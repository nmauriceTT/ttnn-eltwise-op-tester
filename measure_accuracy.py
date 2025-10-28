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



TERM_RED = "\033[91m"
TERM_GREEN = "\033[92m"
TERM_RESET = "\033[0m"

    
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


def measure_op_accuracy_f32(implementations, golden_unary_op, operation_name, dest_dir, group_size=128):
    """
    Measure accuracy of multiple TTNN implementations against a golden reference.
    
    Args:
        implementations: List of (ttnn_op, implementation_name) tuples
        golden_unary_op: Golden reference function
        operation_name: Base operation name (e.g., "tanh", "exp")
        dest_dir: Destination directory for results
        group_size: Number of samples per group for statistics
    """
    
    # Ensure output directory exists
    os.makedirs(f"{dest_dir}/{operation_name}", exist_ok=True)
    
    # Initialize result storage for each implementation
    impl_results = {impl_name: [] for _, impl_name in implementations}
    
    iteration = 0
    for input_tensor in utils.generate_all_f32_tensors():
        print(f"Iteration {iteration}")
        iteration += 1

        # Convert to FP64 for golden function (computed ONCE per iteration)
        torch_input_fp64 = input_tensor.to(torch.float64)
        
        # Run golden operation (computed ONCE per iteration for all implementations)
        golden_torch_fp64 = torch.zeros_like(input_tensor, dtype=torch.float64)
        golden_torch_fp64 = golden_unary_op(torch_input_fp64, out=golden_torch_fp64)

        # Create TTNN input (reused for all implementations)
        ttnn_input = ttnn.from_torch(input_tensor, device=device, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT)

        # Process each implementation
        for ttnn_unary_op, implementation_name in implementations:
            ttnn_output = ttnn.zeros_like(ttnn_input)
            calculated_ttnn_fp32 = ttnn_unary_op(ttnn_input, output_tensor=ttnn_output)

            accuracy_df = compare_with_golden(input_tensor, golden_torch_fp64, calculated_ttnn_fp32, group_size)
            accuracy_df["operation"] = implementation_name
            accuracy_df["dtype"] = "float32"

            impl_results[implementation_name].append(accuracy_df)

    # Save results for each implementation with new naming pattern: {implementation_name}[{group_size}]-{dtype}.csv
    for _, implementation_name in implementations:
        all_df = pd.concat(impl_results[implementation_name])
        all_df.to_csv(f"{dest_dir}/{operation_name}/{implementation_name}-float32-[{group_size}].csv", na_rep="NaN", index_label="index")
        print(f"Saved results for {implementation_name} [float32]")



def measure_op_accuracy_bf16(implementations, golden_unary_op, operation_name, dest_dir, group_size=None):
    """
    Measure accuracy of multiple TTNN implementations against a golden reference.
    
    Args:
        implementations: List of (ttnn_op, implementation_name) tuples
        golden_unary_op: Golden reference function
        operation_name: Base operation name (e.g., "tanh", "exp")
        dest_dir: Destination directory for results
        group_size: Number of samples per group for statistics
    """

    # Ensure group_size is a power of 2
    if group_size is not None and (group_size & (group_size - 1)) != 0:
        raise ValueError(f"Number of samples ({group_size}) must be a power of 2")

    # Create 2^9 x 2^7 tensor (2^16 elements total)
    TENSOR_WIDTH = 2**7
    TENSOR_HEIGHT = 2**9
    size = [TENSOR_HEIGHT, TENSOR_WIDTH]


    # Group by exponent if group_size is not specified (default: 512 = 2^9)
    if group_size is None:
        group_size = 2**9

    start_time = time.time()

    # Create input tensors (computed ONCE)
    input_np = np.arange(0, 2**16, dtype=np.uint32).astype(np.uint16)  # All possible bfloat16 values
    torch_value = torch.from_numpy(input_np).reshape(size)
    torch_input_bf16 = torch_value.view(torch.bfloat16)  # reinterpret data as bfloat16
    torch_input_f64 = torch_input_bf16.to(torch.float64)  # Convert to float64 for torch golden function

    # Run golden operation (computed ONCE for all implementations)
    torch_output_ref = torch.zeros(size, dtype=torch.float64)
    torch_golden_f64 = golden_unary_op(torch_input_f64, out=torch_output_ref)

    # Create TTNN input (reused for all implementations)
    ttnn_input = ttnn.from_torch(torch_input_bf16, device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)

    # Precompute data for PCC calculation (used by all implementations)
    np_flat_input = torch_input_f64.flatten().numpy()
    np_flat_golden = torch_golden_f64.flatten().numpy()

    # Ensure output directory exists
    os.makedirs(f"{dest_dir}/{operation_name}", exist_ok=True)

    # Process each implementation
    for ttnn_unary_op, implementation_name in implementations:
        impl_start_time = time.time()
        
        # Run TTNN operation
        ttnn_output = ttnn.zeros(size, dtype=ttnn.bfloat16, device=device, layout=ttnn.TILE_LAYOUT)
        calculated_ttnn_bf16 = ttnn_unary_op(ttnn_input, output_tensor=ttnn_output)

        # Use compare_with_golden to compute error metrics
        accuracy_df = compare_with_golden(torch_input_f64, torch_golden_f64, calculated_ttnn_bf16, group_size)
        accuracy_df["operation"] = implementation_name
        accuracy_df["dtype"] = "bfloat16"

        # Compute additional statistics (PCC, etc.) for reporting
        torch_ttnn_output_bf16 = ttnn.to_torch(calculated_ttnn_bf16)
        torch_ttnn_output_f64 = torch_ttnn_output_bf16.to(torch.float64)

        np_flat_output = torch_ttnn_output_f64.flatten().numpy()

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

        # Save results with new naming pattern: {implementation_name}[{group_size}]-{dtype}.csv
        accuracy_df.to_csv(f"{dest_dir}/{operation_name}/{implementation_name}-bfloat16-[{group_size}].csv", na_rep="NaN", index_label="index")

        impl_end_time = time.time()
        impl_elapsed_s = impl_end_time - impl_start_time
        print(f"{implementation_name} [bfloat16] PCC = {pcc[0]}, Duration = {impl_elapsed_s:.4f}s")

    end_time = time.time()
    elapsed_s = end_time - start_time
    print(f"Total time for {operation_name} ({len(implementations)} implementations): {elapsed_s:.4f}s")


def parse_operations_config_file(config_file):
    with open(config_file, "r") as f:
        config = json.load(f)
    return config["operations"]


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


    if args.group_size is None:
        if args.type == "bfloat16":
            group_size = 1
        elif args.type == "float32":
            group_size = 65536
        else:
            raise ValueError(f"Invalid data type: {args.type}")
    else:
        group_size = args.group_size

    success_count = 0
    successfull_operations = []
    failed_operations = []

    group_cnt = 0
    total_group_cnt = 0
    

    all_operations = UNARY_OPERATIONS
    for operation_name, op_data in all_operations.items():

        if args.operation is not None and operation_name != args.operation:
            continue

        golden_op = get_golden_function(UNARY_OPERATIONS, operation_name)
        implementations = [(ttnn_op_impl, impl_name) for impl_name, ttnn_op_impl in op_data["implementations"].items()]

        group_cnt += 1
        impl_count = len(implementations)
        print(f"\nProcessing group {group_cnt}/{total_group_cnt}: {operation_name} ({impl_count} implementation(s))")
        
        try:
            start_time = time.time()
            if args.type == "bfloat16":
                measure_op_accuracy_bf16(implementations, golden_op, operation_name, dest_dir, group_size=group_size)
            elif args.type == "float32":
                measure_op_accuracy_f32(implementations, golden_op, operation_name, dest_dir, group_size=group_size)
            else:
                raise ValueError(f"Invalid data type: {args.type}")

            end_time = time.time()
            elapsed_s = end_time - start_time
            print(f"Group {operation_name} completed in {elapsed_s:.4f}s")

            success_count += impl_count
            successfull_operations.extend([impl_name for _, impl_name in implementations])
        except Exception as e:
            logger.warning(f"Could not run operation group {operation_name}: {e}")
            logger.warning(f"{traceback.format_exc()}")
            failed_operations.extend([impl_name for _, impl_name in implementations])

    print(f"\nSucessfully ran {success_count} / {len(all_operations)} operations")
    print(f"{TERM_GREEN}SUCCESS: {successfull_operations}{TERM_RESET}")
    logger.warning(f"FAILED: {failed_operations}")


args = sys.argv
main(args)


ttnn.close_device(device)
