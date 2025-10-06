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

import utils
from models.common.utility_functions import ulp

from arg_parser import parse_args


from operations import UNARY_OPERATIONS

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


# Add powers of [-1, 2, 3, 4, 5, 6, 7, 8, 9, 10] into dictionary
exponents = [2, 3, 4, 5, 6, 7, 8, 9, 10]

operations_dict = UNARY_OPERATIONS

# a**x functions
for exponent in exponents:
    operations_dict[f"pow_{exponent}"] = (
        lambda x, out, e=exponent: torch.pow(e, x),
        lambda x, output_tensor, e=exponent: ttnn.pow(e, x, output_tensor=output_tensor),
        None,
        "pow",
    )
    operations_dict[f"pow21f_{exponent}"] = (
        lambda x, out, e=exponent: torch.pow(e, x),
        lambda x, output_tensor, e=exponent: ttnn.pow(e, x, output_tensor=output_tensor),
        None,
        "pow",
    )

operations_dict["pow21f_tiny"] = (
    lambda x, out, e=0.000001: torch.pow(e, x),
    lambda x, output_tensor, e=0.000001: ttnn.pow(e, x, output_tensor=output_tensor),
    None,
    "pow",
)

# x**a functions
powers = [0, 0.5, 1, 2, 5, 7, 10]
for power in powers:
    operations_dict[f"pow(x,{power})"] = (
        lambda x, out, p=power: torch.pow(x, p),
        lambda x, output_tensor, p=power: ttnn.pow(x, p, output_tensor=output_tensor),
        None,
        "pow",
    )
    operations_dict[f"pow21f(x,{power})"] = (
        lambda x, out, p=power: torch.pow(x, p),
        lambda x, output_tensor, p=power: ttnn.pow(x, p, output_tensor=output_tensor),
        None,
        "pow",
    )


class Measurements:

    def __init__(self):
        pass

    
def compare_with_golden(torch_input: torch.Tensor, golden_torch: torch.Tensor, calculated_ttnn: ttnn.Tensor, group_size: int):

    # Move ttnn to torch
    calculated_torch = ttnn.to_torch(calculated_ttnn)

    # Convert torch output to ttnn dtype for ulp computation
    golden_downcast = golden_torch.to(calculated_torch.dtype)
    golden_ulp = ulp(golden_downcast)

    golden_np_fp64 = golden_torch.to(torch.float64).flatten().numpy()
    calculated_np_fp64 = calculated_torch.to(torch.float64).flatten().numpy()

    EPSILON = 2**-9

    abs_error_np = np.abs(golden_np_fp64 - calculated_np_fp64)
    rel_error_np = abs_error_np / np.maximum(np.abs(golden_np_fp64), EPSILON)
    ulp_error_np = abs_error_np / golden_ulp.flatten().numpy()

    np_input = torch_input.flatten().numpy()

    sub_batches = torch_input.size(0) // group_size

    # Initialize arrays for measurements
    [
        x_array,
        y_array,
        yref_array,
        max_abs_error_array,
        mean_abs_error_array,
        max_ulp_error_array,
        mean_ulp_error_array,
        max_rel_error_array,
        mean_rel_error_array,
    ] = [np.zeros([sub_batches], dtype=np.float64) for _ in range(9)]

    # Process each sub-batch
    for j in range(sub_batches):

        beg_index = j * group_size
        end_index = (j + 1) * group_size

        sub_abs_error_np = abs_error_np[beg_index:end_index]
        sub_rel_error_np = rel_error_np[beg_index:end_index]
        sub_ulp_error_np = ulp_error_np[beg_index:end_index]
        sub_input_np = np_input[beg_index:end_index]
        sub_output_np = calculated_np_fp64[beg_index:end_index]
        sub_ref_np = golden_np_fp64[beg_index:end_index]


        finite_mask = np.isfinite(sub_abs_error_np)
        if np.any(finite_mask):
            max_abs_error = np.max(sub_abs_error_np[finite_mask])
            mean_abs_error = np.mean(sub_abs_error_np[finite_mask])
            max_ulp_error = np.max(sub_ulp_error_np[finite_mask])
            mean_ulp_error = np.mean(sub_ulp_error_np[finite_mask])
            max_rel_error = np.max(sub_rel_error_np[finite_mask])
            mean_rel_error = np.mean(sub_rel_error_np[finite_mask])
        else:
            max_abs_error = np.max(sub_abs_error_np)
            mean_abs_error = np.mean(sub_abs_error_np)
            max_ulp_error = np.max(sub_ulp_error_np)
            mean_ulp_error = np.mean(sub_ulp_error_np)
            max_rel_error = np.max(sub_rel_error_np)
            mean_rel_error = np.mean(sub_rel_error_np)

        # Store results for current sub-batch
        x_array[j] = sub_input_np[0].item()
        y_array[j] = sub_output_np[0].item()
        yref_array[j] = sub_ref_np[0].item()
        max_abs_error_array[j] = max_abs_error.item()
        mean_abs_error_array[j] = mean_abs_error.item()
        max_ulp_error_array[j] = max_ulp_error.item()
        mean_ulp_error_array[j] = mean_ulp_error.item()
        max_rel_error_array[j] = max_rel_error.item()
        mean_rel_error_array[j] = mean_rel_error.item()

    accuracy_df = pd.DataFrame(
        {
            "x": x_array,
            "y": y_array,
            "yref": yref_array,
            "max_abs_error": max_abs_error_array,
            "mean_abs_error": mean_abs_error_array,
            "max_ulp_error": max_ulp_error_array,
            "mean_ulp_error": mean_ulp_error_array,
            "max_rel_error": max_rel_error_array,
            "mean_rel_error": mean_rel_error_array,
        }
    )

    return accuracy_df


def measure_op_accuracy_f32(operation_name, dest_dir, group_size=128):

    (torch_unary_op, ttnn_unary_op, python_unary_op, parent_op) = operations_dict[operation_name]

    all_df = []
    iteration = 0
    for input_tensor in utils.generate_all_f32_tensors():

        print(f"Iteration {iteration}")
        iteration += 1

        ttnn_input = ttnn.from_torch(input_tensor, device=device, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT)

        
        golden_torch_fp64 = torch.zeros_like(input_tensor, dtype=torch.float64)
        golden_torch_fp64 = torch_unary_op(input_tensor, out=golden_torch_fp64)

        ttnn_output = ttnn.zeros_like(ttnn_input)
        calculated_ttnn_fp32 = ttnn_unary_op(ttnn_input, output_tensor=ttnn_output)

        accuracy_df = compare_with_golden(input_tensor,golden_torch_fp64, calculated_ttnn_fp32, group_size)

        all_df.append(accuracy_df)

    all_df = pd.concat(all_df)
    all_df.to_csv(f"{dest_dir}/{operation_name}-f32.csv", na_rep="NaN", index_label="index")


def measure_op_accuracy(operation_name, target_dtype, dest_dir, samples=None):
    parameters = datatypes_parameters[target_dtype]

    # For each tensor, pick several values:
    # e.g. for sub_batch=4, pick, for each i (2**i, 2**i + 2**i/4, 2**i + 2**i/2, 2**i + 3*2**i/4)
    # Within the tensor (2**11, 2**13), these would be indices [(0, 0), (2**9, 0), (2**10), (2**10 + 2**10, 0)]
    # Or, with a general formula: [(0, 0), (TENSOR_HEIGHT//4, 0), (TENSOR_HEIGHT//2, 0), (3*TENSOR_HEIGHT//4, 0)]
    sub_batches = 4
    if samples is not None:
        sub_batches = samples  # TODO: Compute sub_batches from samples instead of just copying data

    TENSOR_WIDTH = parameters["tensor_width"]
    TENSOR_HEIGHT = parameters["tensor_height"]

    SIGN_BITS = parameters["sign_bits"]  # should be 1
    EXPONENT_BITS = parameters["exponent_bits"]
    MANTISSA_BITS = parameters["mantissa_bits"]

    NUMPY_TYPE = parameters["numpy_type"]
    NUMPY_INT_TYPE = parameters["numpy_int_type"]
    TORCH_TYPE = parameters["torch_type"]
    TORCH_INT_TYPE = parameters["torch_int_type"]
    TTNN_TYPE = parameters["ttnn_type"]

    # Tile layout seem to be the main ttnn data layout
    # We could keep data 1D, but with Tile layout, tiles would mostly contain padding data
    # By having 2D tensors, we maximize the filling of each tile
    size = [TENSOR_HEIGHT, TENSOR_WIDTH]

    repeats = 2**EXPONENT_BITS * 2**SIGN_BITS  # sign + exp

    # Use integer => we build floating point numbers exhaustively using their bianry representation
    input_np = np.arange(0, 2**MANTISSA_BITS, dtype=NUMPY_INT_TYPE)

    (host_dtype, dev_dtype) = (TORCH_TYPE, TTNN_TYPE)

    # Create input tensors
    torch_mantissa = torch.from_numpy(input_np).reshape(size)

    torch_exponent = torch.zeros(size, dtype=TORCH_INT_TYPE)
    torch_value = torch.zeros(size, dtype=TORCH_INT_TYPE)
    torch_output_ref = torch.zeros(size, dtype=TORCH_TYPE)
    ttnn_output = ttnn.zeros(size, dtype=TTNN_TYPE, device=device, layout=ttnn.TILE_LAYOUT)

    mse_loss = torch.nn.MSELoss()

    start_time = time.time()

    # Define operations to run

    (torch_unary_op, ttnn_unary_op, python_unary_op, parent_op) = operations_dict[operation_name]

    # Measurements

    [
        x_array,
        y_array,
        yref_array,
        mse_array,
        max_abs_error_array,
        mean_abs_error_array,
        max_rel_error_array,
        mean_rel_error_array,
    ] = [np.zeros([repeats * sub_batches], dtype=NUMPY_TYPE) for _ in range(8)]

    for i in range(0, repeats):
        print(f"{operation_name} [{target_dtype}] iteration #{i} / {repeats}", end="\r")

        # Compute exponent / bit using integer operations
        # Here, we build the binary representation of a set of floating point numbers
        # With this pattern, we have a tensor that contains a set of contiguous floating point numbers
        # All floating point numbers (`torch_input_f32`) will share the same exponent & sign but will have unique mantissa

        torch.full(size, i, dtype=TORCH_INT_TYPE, out=torch_exponent)  # exp bits = i
        torch.bitwise_left_shift(torch_exponent, MANTISSA_BITS, out=torch_exponent)
        torch.bitwise_or(torch_exponent, torch_mantissa, out=torch_value)  # combine sign/exponent with mantissa

        torch_input_f32 = torch_value.view(TORCH_TYPE)  # reinterpret data as float32

        # print(f"Torch Value =\n{torch_value}, size = {torch_value.size()}")
        # print(f"Torch Value f32 =\n{torch_value_f32}, size = {torch_value_f32.size()}")

        # Launch a TTNN operation from a torch tensor and returns output in torch tensor
        def launch_ttnn_op(torch_tensor, ttnn_unary, ttnn_output):
            ttnn_value = ttnn.from_torch(torch_tensor, device=device, dtype=TTNN_TYPE, layout=ttnn.TILE_LAYOUT)

            ttnn_output = ttnn_unary(ttnn_value, output_tensor=ttnn_output)

            # Convert back to torch
            torch_output = ttnn.to_torch(ttnn_output)

            return torch_output

        def launch_scalar_op(torch_tensor, python_unary):
            np_input = torch_tensor.to(torch.float32).flatten().numpy()

            def run_unary_op(x):
                try:
                    return python_unary(x)
                except:
                    return math.nan

            np_output = np.vectorize(run_unary_op)(np_input)

            torch_output = torch.from_numpy(np_output).to(TORCH_TYPE).reshape(size)
            return torch_output

        # Run operation
        torch_output_ref = torch_unary_op(torch_input_f32, out=torch_output_ref)

        if True:
            actual_torch_output = launch_ttnn_op(torch_input_f32, ttnn_unary_op, ttnn_output)
        else:  # Launch scalar op (used to evaluate accuracy of torch)
            actual_torch_output = launch_scalar_op(torch_input_f32, python_unary_op)

        # Flatten tensors for data analysis (we only used 2D for ttnn and TILE_LAYOUT)
        np_flat_input = torch_input_f32.to(torch.float32).flatten().numpy()
        np_flat_output = actual_torch_output.to(torch.float32).flatten().numpy()
        np_flat_ref = torch_output_ref.to(torch.float32).flatten().numpy()
        # np_flat_exponent = torch_exponent.flatten().view(TORCH_TYPE).numpy()

        # TODO: Just switch to pandas and do groupby() & cie
        for j in range(0, sub_batches):
            chunk_size = TENSOR_WIDTH * TENSOR_HEIGHT // sub_batches
            (beg_index, end_index) = (j * chunk_size, (j + 1) * chunk_size)
            res_i = i * sub_batches + j

            # TODO: Handle NaN/inf

            # Get sub-range
            np_sub_input = np_flat_input[beg_index:end_index]
            np_sub_output = np_flat_output[beg_index:end_index]
            np_sub_ref = np_flat_ref[beg_index:end_index]

            # Measure abs error
            np_diff = np.abs(np_sub_ref - np_sub_output)

            # Compare actual and expected output
            # mse_value      = mse_loss(actual_sub_output, torch_sub_ref)
            np_diff_curated = np_diff[~np.isfinite(np_diff)]
            np_sub_ref_abs = np.abs(np_sub_ref[~np.isfinite(np_sub_ref)])

            if len(np_diff) > 0 and len(np_sub_ref_abs) > 0:
                # Reduces edge cases
                max_abs_error = np_diff_curated.max()
                mean_abs_error = np_diff_curated.mean()
                rel_error = np_diff_curated / max(np_sub_ref_abs.max(), EPSILON)
                max_rel_error = np.max(rel_error)  # Ignore NaN
                mean_rel_error = np.mean(rel_error)

            else:  # Batch only contains infinite value
                max_abs_error = np_diff.max()
                mean_abs_error = np_diff.mean()
                max_rel_error = np.max(np_diff / np.abs(np_sub_ref))  # Ignore NaN
                mean_rel_error = np.mean(np_diff / np.abs(np_sub_ref))

            # Write output data at given sub-batches / sub-samples
            x_array[res_i] = np_sub_input[0].item()
            y_array[res_i] = np_sub_output[0].item()
            yref_array[res_i] = np_sub_ref[0].item()
            # mse_array           [res_i] = mse_value.item()
            max_abs_error_array[res_i] = max_abs_error.item()
            mean_abs_error_array[res_i] = mean_abs_error.item()
            max_rel_error_array[res_i] = max_rel_error.item()
            mean_rel_error_array[res_i] = mean_rel_error.item()

    accuracy_df = pd.DataFrame(
        {
            "base_x": x_array,
            "base_y": y_array,
            "base_yref": yref_array,
            "mse": mse_array,
            "max_abs_error": max_abs_error_array,
            "mean_abs_error": mean_abs_error_array,
            "max_rel_error": max_rel_error_array,
            "mean_rel_error": mean_rel_error_array,
        }
    )
    accuracy_df["operation"] = operation_name
    accuracy_df["dtype"] = target_dtype

    accuracy_df.to_csv(f"{dest_dir}/{operation_name}-{target_dtype}-[{samples}].csv", na_rep="NaN", index_label="index")

    end_time = time.time()
    elapsed_s = end_time - start_time
    elapsed_ms = (elapsed_s) * 1000
    print(f"Duration = {elapsed_s}s, {elapsed_ms/repeats} ms/iteration")


def measure_op_accuracy_bf16(operation_name, dest_dir, group_size=None):
    # Use bfloat16 parameters
    parameters = datatypes_parameters["bfloat16"]

    # Ensure group_size is a power of 2
    if group_size is not None and (group_size & (group_size - 1)) != 0:
        raise ValueError(f"Number of samples ({group_size}) must be a power of 2")

    # Create 2^9 x 2^7 tensor (2^16 elements total)
    TENSOR_WIDTH = 2**7
    TENSOR_HEIGHT = 2**9
    size = [TENSOR_HEIGHT, TENSOR_WIDTH]

    SIGN_BITS = parameters["sign_bits"]  # should be 1
    EXPONENT_BITS = parameters["exponent_bits"]
    MANTISSA_BITS = parameters["mantissa_bits"]

    NUMPY_TYPE = parameters["numpy_type"]
    NUMPY_INT_TYPE = parameters["numpy_int_type"]
    TORCH_TYPE = parameters["torch_type"]
    TORCH_INT_TYPE = parameters["torch_int_type"]
    TTNN_TYPE = parameters["ttnn_type"]

    # Group by exponent if group_size is not specified
    sub_batches = 2**9 if group_size is None else 2**16 // group_size

    # Create input tensors
    input_np = np.arange(0, 2**16, dtype=NUMPY_INT_TYPE)  # All possible bfloat16 values
    torch_value = torch.from_numpy(input_np).reshape(size)
    torch_input_bf16 = torch_value.view(TORCH_TYPE)  # reinterpret data as bfloat16
    torch_input_f64 = torch_input_bf16.to(torch.float64)  # Convert to float64 for torch golden function

    torch_output_ref = torch.zeros(size, dtype=torch.float64)
    ttnn_output = ttnn.zeros(size, dtype=TTNN_TYPE, device=device, layout=ttnn.TILE_LAYOUT)

    # Get the operations to test
    (torch_unary_op, ttnn_unary_op, python_unary_op, parent_op) = operations_dict[operation_name]

    # Initialize arrays for measurements
    [
        x_array,
        y_array,
        yref_array,
        max_abs_error_array,
        mean_abs_error_array,
        max_ulp_error_array,
        mean_ulp_error_array,
        max_rel_error_array,
        mean_rel_error_array,
    ] = [np.zeros([sub_batches], dtype=np.float64) for _ in range(9)]

    start_time = time.time()

    # Launch TTNN operation
    def launch_ttnn_op(torch_tensor, ttnn_unary, ttnn_output):
        ttnn_value = ttnn.from_torch(torch_tensor, device=device, dtype=TTNN_TYPE, layout=ttnn.TILE_LAYOUT)
        ttnn_output = ttnn_unary(ttnn_value, output_tensor=ttnn_output)
        return ttnn.to_torch(ttnn_output)

    # Run reference and actual operations
    torch_golden_f64 = torch_unary_op(torch_input_f64, out=torch_output_ref)
    torch_ttnn_output_bf16 = launch_ttnn_op(torch_input_bf16, ttnn_unary_op, ttnn_output)

    torch_golden_bf16 = torch_golden_f64.to(torch.bfloat16)
    torch_ttnn_output_f64 = torch_ttnn_output_bf16.to(torch.float64)

    # Compute errors
    np_golden_f64 = torch_golden_f64.flatten().numpy()
    np_ttnn_output_f64 = torch_ttnn_output_f64.flatten().numpy()
    np_diff = np.abs(np_golden_f64 - np_ttnn_output_f64)

    golden_ulp = ulp(torch_golden_bf16).to(torch.float64)
    ulp_delta = np_diff / golden_ulp.flatten().numpy()

    torch_ulp_value = utils.ulp_bf16(torch_golden_bf16).to(torch.float64)
    torch_eps = torch.full(torch_input_bf16.size(), EPSILON, dtype=torch.float64)
    np_eps = np.full(2**16, EPSILON)

    np_rel_error = np_diff / np.maximum(np.abs(np_golden_f64), np_eps)
    np_ulp_error = np_diff / torch_ulp_value.flatten().numpy()
    np_ulp_error = ulp_delta

    finite_mask = np.isfinite(np_golden_f64) & np.isfinite(np_ttnn_output_f64)  # Don't compute PCC on NaN and infinity
    pcc = scipy.stats.pearsonr(np_golden_f64[finite_mask], np_ttnn_output_f64[finite_mask])

    # Flatten tensors and convert to ndarray for analysis
    np_flat_input = torch_input_f64.flatten().numpy()
    np_flat_output = torch_ttnn_output_f64.flatten().numpy()
    np_flat_golden = torch_golden_f64.flatten().numpy()

    # Process each sub-batch
    for j in range(0, sub_batches):
        chunk_size = TENSOR_WIDTH * TENSOR_HEIGHT // sub_batches
        (beg_index, end_index) = (j * chunk_size, (j + 1) * chunk_size)

        # Get sub-range
        np_sub_input = np_flat_input[beg_index:end_index]
        np_sub_output = np_flat_output[beg_index:end_index]
        np_sub_ref = np_flat_golden[beg_index:end_index]
        np_sub_diff = np_diff[beg_index:end_index]
        np_sub_rel_error = np_rel_error[beg_index:end_index]
        np_sub_ulp_error = np_ulp_error[beg_index:end_index]

        # Calculate errors

        finite_mask = np.isfinite(np_sub_diff)
        if np.any(finite_mask):
            max_abs_error = np.max(np_sub_diff[finite_mask])
            mean_abs_error = np.mean(np_sub_diff[finite_mask])

            max_ulp_error = np.max(np_sub_ulp_error[finite_mask])
            mean_ulp_error = np.mean(np_sub_ulp_error[finite_mask])

            max_rel_error = np.max(np_sub_rel_error[finite_mask])
            mean_rel_error = np.mean(np_sub_rel_error[finite_mask])

        else:
            max_abs_error = np.max(np_sub_diff)
            mean_abs_error = np.mean(np_sub_diff)

            max_ulp_error = np.max(np_sub_ulp_error)
            mean_ulp_error = np.mean(np_sub_ulp_error)

            max_rel_error = np.max(np_sub_rel_error)
            mean_rel_error = np.mean(np_sub_rel_error)

        # Store results
        x_array[j] = np_sub_input[0].item()
        y_array[j] = np_sub_output[0].item()
        yref_array[j] = np_sub_ref[0].item()
        max_abs_error_array[j] = max_abs_error.item()
        mean_abs_error_array[j] = mean_abs_error.item()
        max_ulp_error_array[j] = max_ulp_error.item()
        mean_ulp_error_array[j] = mean_ulp_error.item()
        max_rel_error_array[j] = max_rel_error.item()
        mean_rel_error_array[j] = mean_rel_error.item()

    # Create and save DataFrame
    accuracy_df = pd.DataFrame(
        {
            "base_x": x_array,
            "base_y": y_array,
            "base_yref": yref_array,
            "max_abs_error": max_abs_error_array,
            "mean_abs_error": mean_abs_error_array,
            "max_ulp_error": max_ulp_error_array,
            "mean_ulp_error": mean_ulp_error_array,
            "max_rel_error": max_rel_error_array,
            "mean_rel_error": mean_rel_error_array,
        }
    )
    accuracy_df["operation"] = operation_name
    accuracy_df["dtype"] = "bfloat16"
    accuracy_df["parent_op"] = parent_op

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

    golden_std = np.std(np_flat_golden)
    ttnn_std = np.std(np_flat_output)

    np_finite_ulp_mask = np.isfinite(np_ulp_error) & (
        np.greater(np_flat_input, -(2**6)) & np.less(np_flat_input, 2**6)
    )
    mean_ulp_error = np.mean(np_ulp_error[np_finite_ulp_mask])
    print(f"Finite ulp error = {np_ulp_error[np_finite_ulp_mask]}")

    print(f"Mean ulp error = {mean_ulp_error}")

    covar = np.cov(np_flat_golden, np_flat_output)
    print(f"Golden std = {golden_std}, TTNN std = {ttnn_std}")
    print(f"Covar = {covar}")

    accuracy_df.to_csv(f"{dest_dir}/{operation_name}-bfloat16-[{group_size}].csv", na_rep="NaN", index_label="index")

    end_time = time.time()
    elapsed_s = end_time - start_time
    print(f"{operation_name} [bfloat16] PCC = {pcc[0]}, Duration = {elapsed_s:.4f}s")


def parse_operations_config_file(config_file):
    with open(config_file, "r") as f:
        config = json.load(f)
    return config["operations"]


def main(args):

    args = parse_args("unary")
    group_size = args.group_size


    dest_dir = "accuracy_results/results/"
    if not os.path.exists(dest_dir):  # TODO: Check if recursive
        os.makedirs(dest_dir)

    # Set numpy floating point warning to reduce stdout clutter
    # Since we test *all* possible floating point values, invalid values
    # are expected.
    # TODO: Log warnings into file
    np.seterr(divide="ignore")
    np.seterr(invalid="ignore")
    np.seterr(over="ignore")

    # Add powers into operations
    powers_vals = [2, 3, 4, 5, 6, 7, 8, 9, 10]
    powers = [f"pow_{power}" for power in powers_vals]


    if args.operation is not None:
        all_operations = [args.operation]
    else:
        all_operations = parse_operations_config_file("op_configs/unary_operations.json")


    # all_operations += powers
    # highres_operations += powers

    success_count = 0
    successfull_operations = []
    failed_operations = []

    cnt = 0
    total_operation_cnt = len(all_operations)
    print(f"Measuring operations")
    for operation in all_operations:
        cnt += 1
        print(f"Running operation {operation} #{cnt}/{total_operation_cnt}", end="\r")
        try:

            start_time = time.time()
            if args.type == "bfloat16":
                measure_op_accuracy_bf16(operation, dest_dir, group_size=group_size)
            elif args.type == "float32":
                measure_op_accuracy_f32(operation, dest_dir, group_size=group_size)
            else:
                raise ValueError(f"Invalid data type: {args.type}")

            end_time = time.time()
            elapsed_s = end_time - start_time
            print(f"Duration = {elapsed_s}s")

            success_count += 1
            successfull_operations += [f"{operation}"]
        except Exception as e:
            print(f"{TERM_RED}Could not run operation {operation}: {e}{TERM_RESET}")
            print(f"{TERM_RED}{traceback.format_exc()}{TERM_RESET}")
            failed_operations += [f"{operation}"]

    print(f"Sucessfully ran {success_count} / {len(all_operations)} operations")
    print(f"{TERM_GREEN}SUCCESS: {successfull_operations}{TERM_RESET}")
    print(f"{TERM_RED}FAILED: {failed_operations}{TERM_RESET}")


args = sys.argv
main(args)


ttnn.close_device(device)
