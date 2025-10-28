import ttnn
import torch
import math
import numpy as np
import time
import pandas as pd
import sys
import os
import traceback
import scipy
import json
from loguru import logger

import utils
from models.common.utility_functions import ulp


from arg_parser import parse_args
from operations import UNARY_OPERATIONS, iterate_all_operations, get_operation_by_name, get_golden_function
from kernel_generator import generate_unary_kernel_from_polynomial, generate_unary_kernel_from_sfpi_source


device_id = 0
device = ttnn.open_device(device_id=device_id)

EPSILON = 2**-9


# ttnn.enable_program_cache(device)  # Useful: we are going to call the same kernel several times


datatypes_parameters = {
    "float32": {
        "numpy_type": np.float32,
        "torch_type": torch.float32,
        "ttnn_type": ttnn.float32,
        "numpy_int_type": np.int32,
        "torch_int_type": torch.int32,
        "sign_bits": 1,
        "exponent_bits": 8,
        "mantissa_bits": 23,
        "tensor_width": 2**12,
        "tensor_height": 2**11,
    },
    "bfloat16": {
        "numpy_type": np.float32,  # Note: a conversion will be needed
        "torch_type": torch.bfloat16,
        "ttnn_type": ttnn.bfloat16,
        "numpy_int_type": np.int16,
        "torch_int_type": torch.int16,
        "sign_bits": 1,
        "exponent_bits": 8,
        "mantissa_bits": 7,
        "tensor_width": 2**4,  # Not great (< 32) => tiles will have padding
        "tensor_height": 2**3,
    },
}


TERM_RED = "\033[91m"
TERM_GREEN = "\033[92m"
TERM_RESET = "\033[0m"




class Measurements:

    def __init__(self):
        pass

    
def compare_with_golden(torch_input: torch.Tensor, golden_torch: torch.Tensor, calculated_ttnn: ttnn.Tensor, group_size: int):


    with np.testing.suppress_warnings() as sup:
        # Avoid warnings such as "Mean of empty slice"
        # slice full of NaNs, etc..
        # These situations can happen when testing exhaustively, and should be ignored
        sup.filter(RuntimeWarning, "")

        # Data type used for compute mean/max (more compact than float64 to speed-up computation)
        np_compute_dtype = np.float32

        # Move ttnn to torch
        calculated_torch = ttnn.to_torch(calculated_ttnn)

        # Convert torch output to ttnn dtype for ulp computation
        golden_downcast = golden_torch.to(calculated_torch.dtype)
        golden_ulp = ulp(golden_downcast).to(golden_torch.dtype)


        torch_input_size = torch_input.nelement()
        sub_batches = torch_input_size // group_size

        measurement_shape = [sub_batches, group_size]

        golden_np_fp64 = golden_torch.to(torch.float64).flatten().numpy().reshape(measurement_shape)
        calculated_np_fp64 = calculated_torch.to(torch.float64).flatten().numpy().reshape(measurement_shape)
        np_input = torch_input.flatten().numpy().reshape(measurement_shape)

        EPSILON = 2**-9


        abs_error_np = np.abs(golden_np_fp64 - calculated_np_fp64).astype(np_compute_dtype)
        rel_error_np = abs_error_np / np.maximum(np.abs(golden_np_fp64), EPSILON).astype(np_compute_dtype)
        ulp_error_np = abs_error_np / golden_ulp.flatten().numpy().reshape(measurement_shape).astype(np_compute_dtype)

        assert len(abs_error_np) > 0
        # finite_mask = np.isfinite(abs_error_np)
        x_array = np_input.take(0, axis=-1).astype(np_compute_dtype).flatten()
        y_array = calculated_np_fp64.take(0, axis=-1).astype(np_compute_dtype).flatten()
        yref_array = golden_np_fp64.take(0, axis=-1).astype(np_compute_dtype).flatten()
        max_abs_error = np.nanmax(abs_error_np, axis=-1).flatten()
        mean_abs_error = np.nanmean(abs_error_np, axis=-1).flatten()
        max_ulp_error = np.nanmax(ulp_error_np, axis=-1).flatten()
        mean_ulp_error = np.nanmean(ulp_error_np, axis=-1).flatten()
        max_rel_error = np.nanmax(rel_error_np, axis=-1).flatten()
        mean_rel_error = np.nanmean(rel_error_np, axis=-1).flatten()

    accuracy_df = pd.DataFrame(
        {
            "base_x": x_array,
            "base_y": y_array,
            "base_yref": yref_array,
            "max_abs_error": max_abs_error,
            "mean_abs_error": mean_abs_error,
            "max_ulp_error": max_ulp_error,
            "mean_ulp_error": mean_ulp_error,
            "max_rel_error": max_rel_error,
            "mean_rel_error": mean_rel_error,
        }
    )

    return accuracy_df


def measure_op_accuracy_f32(ttnn_unary_op, golden_unary_op, operation_name, dest_dir, group_size=128):

    all_df = []
    iteration = 0
    for input_tensor in utils.generate_all_f32_tensors():

        print(f"Iteration {iteration}")
        iteration += 1

        ttnn_input = ttnn.from_torch(input_tensor, device=device, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT)
        torch_input_fp64 = input_tensor.to(torch.float64)
        
        golden_torch_fp64 = torch.zeros_like(input_tensor, dtype=torch.float64)
        golden_torch_fp64 = golden_unary_op(torch_input_fp64, out=golden_torch_fp64)

        ttnn_output = ttnn.zeros_like(ttnn_input)
        calculated_ttnn_fp32 = ttnn_unary_op(ttnn_input, output_tensor=ttnn_output)

        accuracy_df = compare_with_golden(input_tensor,golden_torch_fp64, calculated_ttnn_fp32, group_size)
        accuracy_df["operation"] = operation_name
        accuracy_df["dtype"] = "float32"

        all_df.append(accuracy_df)

    all_df = pd.concat(all_df)
    os.makedirs(f"{dest_dir}/{operation_name}", exist_ok=True)
    all_df.to_csv(f"{dest_dir}/{operation_name}/{operation_name}-float32-[{group_size}].csv", na_rep="NaN", index_label="index")



def measure_op_accuracy_bf16(ttnn_unary_op, golden_unary_op, operation_name, dest_dir, group_size=None):
    # Use bfloat16 parameters
    parameters = datatypes_parameters["bfloat16"]

    # Ensure group_size is a power of 2
    if group_size is not None and (group_size & (group_size - 1)) != 0:
        raise ValueError(f"Number of samples ({group_size}) must be a power of 2")

    # Create 2^9 x 2^7 tensor (2^16 elements total)
    TENSOR_WIDTH = 2**7
    TENSOR_HEIGHT = 2**9
    size = [TENSOR_HEIGHT, TENSOR_WIDTH]

    NUMPY_INT_TYPE = parameters["numpy_int_type"]
    TORCH_TYPE = parameters["torch_type"]
    TTNN_TYPE = parameters["ttnn_type"]

    # Group by exponent if group_size is not specified (default: 512 = 2^9)
    if group_size is None:
        group_size = 2**9

    start_time = time.time()

    # Create input tensors
    input_np = np.arange(0, 2**16, dtype=NUMPY_INT_TYPE)  # All possible bfloat16 values
    torch_value = torch.from_numpy(input_np).reshape(size)
    torch_input_bf16 = torch_value.view(TORCH_TYPE)  # reinterpret data as bfloat16
    torch_input_f64 = torch_input_bf16.to(torch.float64)  # Convert to float64 for torch golden function

    # Run golden operation
    torch_output_ref = torch.zeros(size, dtype=torch.float64)
    torch_golden_f64 = golden_unary_op(torch_input_f64, out=torch_output_ref)

    # Run TTNN operation
    ttnn_input = ttnn.from_torch(torch_input_bf16, device=device, dtype=TTNN_TYPE, layout=ttnn.TILE_LAYOUT)
    ttnn_output = ttnn.zeros(size, dtype=TTNN_TYPE, device=device, layout=ttnn.TILE_LAYOUT)
    calculated_ttnn_bf16 = ttnn_unary_op(ttnn_input, output_tensor=ttnn_output)

    # Use compare_with_golden to compute error metrics
    accuracy_df = compare_with_golden(torch_input_f64, torch_golden_f64, calculated_ttnn_bf16, group_size)
    accuracy_df["operation"] = operation_name
    accuracy_df["dtype"] = "bfloat16"

    # Compute additional statistics (PCC, etc.) for reporting
    torch_ttnn_output_bf16 = ttnn.to_torch(calculated_ttnn_bf16)
    torch_ttnn_output_f64 = torch_ttnn_output_bf16.to(torch.float64)

    np_flat_input = torch_input_f64.flatten().numpy()
    np_flat_output = torch_ttnn_output_f64.flatten().numpy()
    np_flat_golden = torch_golden_f64.flatten().numpy()

    # Compute PCC on [-1e5; 1e5]
    np_finite_mask = np.isfinite(np_flat_output) & np.isfinite(np_flat_golden)
    df = pd.DataFrame(
        {
            "x": np_flat_input[np_finite_mask],
            "y": np_flat_output[np_finite_mask],
            "yref": np_flat_golden[np_finite_mask],
        }
    )

    df = df[df["x"].between(-1e5, 1e5)]
    pcc = scipy.stats.pearsonr(df["y"], df["yref"])

    # Save results
    os.makedirs(f"{dest_dir}/{operation_name}", exist_ok=True)
    accuracy_df.to_csv(f"{dest_dir}/{operation_name}/{operation_name}-bfloat16-[{group_size}].csv", na_rep="NaN", index_label="index")

    end_time = time.time()
    elapsed_s = end_time - start_time
    print(f"{operation_name} [bfloat16] PCC = {pcc[0]}, Duration = {elapsed_s:.4f}s")


def parse_operations_config_file(config_file):
    with open(config_file, "r") as f:
        config = json.load(f)
    return config["operations"]


def generate_tanh_alternative():

    # tanh_operations = [
    #     "tanh",
    #     "tanh-v1",
    #     "tanh-pade-5,5"
    # ]

    new_operations = {}
    polynomial_expressions = {
        # "tanh-Chebyshev-v1-c0ef0[6]": [0.004613510798662901,-0.0569886788725853,0.25763407349586487,-0.46735504269599915,0.02672632411122322,0.9987236261367798,0.0],
        # "tanh-minimax-v0[4]": [2.49048434197902679443359375e-2, -8.3681561052799224853515625e-2, -0.20078647136688232421875,1.0220668315887451171875, 0.0],
        # "tanh-minimax-v1[5]": [-1.950809545814990997314453125e-2, 0.1467897593975067138671875, -0.325587689876556396484375, -4.27231900393962860107421875e-2, 1.00523841381072998046875, 0.0],
    }

    for op_name, polynomial_coefficients in polynomial_expressions.items():

        new_operations[op_name] = lambda x, output_tensor, ttnn_function=generate_unary_kernel_from_polynomial("tanh-poly", polynomial_coefficients): ttnn_function(x, output_tensor)

    new_operations["tanh-sfpi"] = lambda x, output_tensor, ttnn_function=generate_unary_kernel_from_sfpi_source("tanh"): ttnn_function(x, output_tensor)
    new_operations["tanh-v1"] = lambda x, output_tensor, ttnn_function=generate_unary_kernel_from_sfpi_source("tanh-v1"): ttnn_function(x, output_tensor)
    new_operations["tanh-pade-5,5"] = lambda x, output_tensor, ttnn_function=generate_unary_kernel_from_sfpi_source("tanh-pade-5,5"): ttnn_function(x, output_tensor)
    new_operations["tanh-minimax-v1[6]"] = lambda x, output_tensor, ttnn_function=generate_unary_kernel_from_sfpi_source("tanh-minimax-v1[6]"): ttnn_function(x, output_tensor)


    return new_operations


def generate_exponential_alternative():

    polynomial_expressions = {
        #"exp-Chebyshev-v1[2]": [0.34228965640068054,0.652752697467804,1.0022648572921753],
        "exp-Chebyshev-v1-c0ef0[4]": [0.012763113714754581,0.05344102904200554,0.24064704775810242,0.6931340098381042,1.0],
        # "exp-Chebyshev-v1[4]": [0.013670309446752071,0.05174499750137329,0.24160435795783997,0.6929728984832764,1.000003457069397],
        "exp-61f": [0.0002170391, 0.001243946, 0.0096788315, 0.055483369, 0.24022982, 0.69314699, 1.0000000018],
        "exp-21f": [0.33718944, 0.65763629, 1.0017248],
        "exp-Chebyshev-v1[6]": [0.00021865784947294742,0.0012391331838443875,0.009684186428785324,0.055480629205703735,0.24023045599460602,0.6931469440460205,1.0]
    }

    new_operations = {}

    for op_name, polynomial_coefficients in polynomial_expressions.items():

        ttnn_function = generate_unary_kernel_from_polynomial("exp", polynomial_coefficients, full_name=op_name)

        new_operations[op_name] = lambda x, output_tensor, ttnn_function=ttnn_function: ttnn_function(x, output_tensor)
        

    return new_operations


def generate_sigmoid_alternative():

    new_operations = {}

    new_operations["sigmoid-21f"] = lambda x, output_tensor, ttnn_function=generate_unary_kernel_from_sfpi_source("sigmoid"): ttnn_function(x, output_tensor)

    return new_operations


def generate_log2_alternative():
    new_operations = {}

    # x * (1.43861830234527587890625 + x * (-0.6402385234832763671875 + x * 0.2044459879398345947265625))
    # `x * (1.44261324405670166015625 + x * (-0.7169878482818603515625 + x * (0.441900312900543212890625 + x * (-0.22712136805057525634765625 + x * 5.96523843705654144287109375e-2))))`

    polynomial_expressions = {
        "log2-minimax-v1[3]": [0.2044459879398345947265625, -0.6402385234832763671875, 1.43861830234527587890625, 0.0],
        "log2-minimax-v1[5]": [5.96523843705654144287109375e-2, -0.22712136805057525634765625, 0.441900312900543212890625, -0.7169878482818603515625, 1.44261324405670166015625, 0.0]
    }

    for op_name, polynomial_coefficients in polynomial_expressions.items():
        ttnn_function = generate_unary_kernel_from_polynomial("log2-poly", polynomial_coefficients, full_name=op_name)
        new_operations[op_name] = lambda x, output_tensor, ttnn_function=ttnn_function: ttnn_function(x, output_tensor)

    # new_operations["log2-v0"] = lambda x, output_tensor, ttnn_function=generate_unary_kernel_from_sfpi_source("log2"): ttnn_function(x, output_tensor)

    return new_operations


def main(args):

    args = parse_args("unary")


    dest_dir = "accuracy_results/results/unary/"
    os.makedirs(dest_dir, exist_ok=True)

    # Set numpy floating point warning to reduce stdout clutter
    # Since we test *all* possible floating point values, invalid values
    # are expected.
    # TODO: Log warnings into file
    np.seterr(divide="ignore")
    np.seterr(invalid="ignore")
    np.seterr(over="ignore")


    if args.operation is not None:
        all_operation_names = [args.operation]
    else:
        all_operation_names = parse_operations_config_file("op_configs/unary_operations.json")

    if args.group_size is None:
        if args.type == "bfloat16":
            group_size = 1
        elif args.type == "float32":
            group_size = 65536
        else:
            raise ValueError(f"Invalid data type: {args.type}")
    else:
        group_size = args.group_size


    all_operations = {}
    for operation_name in all_operation_names:

        operation = get_operation_by_name(UNARY_OPERATIONS, operation_name)

        if operation is None:
            logger.warning(f"Operation {operation_name} not found in UNARY_OPERATIONS")
        else:
            all_operations[operation_name] = operation

    if args.kernel is not None:

        operation_name = args.kernel

        extra_operations = {}
        if operation_name == "tanh":
            extra_operations |= generate_tanh_alternative()
        elif operation_name == "exp":
            extra_operations |= generate_exponential_alternative()
        elif operation_name == "sigmoid":
            extra_operations |= generate_sigmoid_alternative()
        elif operation_name == "log2":
            extra_operations |= generate_log2_alternative()
        else:
            raise ValueError(f"Invalid kernel: {operation_name}")

        golden_function = get_golden_function(UNARY_OPERATIONS, operation_name)

        if operation_name in all_operations:

            for extra_operation_name, extra_operation in extra_operations.items():
                all_operations[extra_operation_name] = (extra_operation, golden_function)


    success_count = 0
    successfull_operations = []
    failed_operations = []

    cnt = 0
    total_operation_cnt = len(all_operations)
    print(f"Measuring operations")
    for operation_name, (ttnn_op, golden_op) in all_operations.items():

        cnt += 1
        print(f"Running operation {operation_name} #{cnt}/{total_operation_cnt}", end="\r")
        try:
            start_time = time.time()
            if args.type == "bfloat16":
                measure_op_accuracy_bf16(ttnn_op, golden_op, operation_name, dest_dir, group_size=group_size)
            elif args.type == "float32":
                measure_op_accuracy_f32(ttnn_op, golden_op, operation_name, dest_dir, group_size=group_size)
            else:
                raise ValueError(f"Invalid data type: {args.type}")

            end_time = time.time()
            elapsed_s = end_time - start_time
            print(f"Duration = {elapsed_s}s")

            success_count += 1
            successfull_operations += [f"{operation_name}"]
        except Exception as e:
            logger.warning(f"Could not run operation {operation_name}: {e}")
            logger.warning(f"{traceback.format_exc()}")
            failed_operations += [f"{operation_name}"]

    print(f"Sucessfully ran {success_count} / {len(all_operations)} operations")
    print(f"{TERM_GREEN}SUCCESS: {successfull_operations}{TERM_RESET}")
    logger.warning(f"FAILED: {failed_operations}")


args = sys.argv
main(args)


ttnn.close_device(device)
