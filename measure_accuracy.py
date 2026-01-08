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
from operations import UNARY_OPERATIONS, BINARY_OPERATIONS, iterate_all_operations, get_operation_variant_by_name, get_golden_function, run_ttnn_op


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
    
    # Generate summary
    generate_summary(implementations, golden_unary_op, operation_name, dest_dir, "float32", impl_results_dict=impl_results)



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

    # Ensure output directory exists
    os.makedirs(f"{dest_dir}/{operation_name}", exist_ok=True)

    # Initialize result storage for each implementation
    impl_results = {impl_name: [] for _, impl_name in implementations}

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

        # Store results for summary generation
        impl_results[implementation_name].append(accuracy_df)

        # Save results with new naming pattern: {implementation_name}[{group_size}]-{dtype}.csv
        accuracy_df.to_csv(f"{dest_dir}/{operation_name}/{implementation_name}-bfloat16-[{group_size}].csv", na_rep="NaN", index_label="index")

        impl_end_time = time.time()
        impl_elapsed_s = impl_end_time - impl_start_time
        print(f"{implementation_name} [bfloat16] Duration = {impl_elapsed_s:.4f}s")

    # Generate summary
    generate_summary(implementations, golden_unary_op, operation_name, dest_dir, "bfloat16", impl_results_dict=impl_results)

    end_time = time.time()
    elapsed_s = end_time - start_time
    print(f"Total time for {operation_name} ({len(implementations)} implementations): {elapsed_s:.4f}s")


def parse_operations_config_file(config_file):
    with open(config_file, "r") as f:
        config = json.load(f)
    return config["operations"]


def detect_operation_type(operation_name):
    """Automatically determine if an operation is unary or binary by checking both dictionaries."""
    if operation_name in UNARY_OPERATIONS:
        return "unary"
    elif operation_name in BINARY_OPERATIONS:
        return "binary"
    else:
        raise ValueError(f"Operation '{operation_name}' not found in UNARY_OPERATIONS or BINARY_OPERATIONS")


# Binary operation measurement functions (from measure_binary.py)

def reduce_batch_and_cols(tensor):
    """Take 3D tensor and return 1D tensor
    Input: [batch, rows, cols]
    Output: [rows]
    """
    (tensor_tmp, _) = torch.max(tensor, dim=0)
    (tensor_max, _) = tensor_tmp.max(dim=1)
    return tensor_max


def reduce_on_batch_and_cols(tensor):
    """Reduce 3D tensor across batch and columns dimensions."""
    # Compute max
    (tensor_tmp, _) = torch.max(tensor, dim=0)
    (tensor_max, _) = torch.max(tensor_tmp, dim=1)

    # Compute min
    (tensor_tmp, _) = torch.min(tensor, dim=0)
    (tensor_min, _) = torch.min(tensor_tmp, dim=1)

    # Compute average
    tensor_tmp = torch.mean(tensor, dim=0)
    tensor_mean = torch.mean(tensor_tmp, dim=1)

    return {"min": tensor_min, "max": tensor_max, "mean": tensor_mean}


def flush_subnormals(tensor, min_normal_value=2**-126):
    """Remove data where inputs are subnormals."""
    return torch.where(tensor.abs() >= min_normal_value, tensor, torch.zeros_like(tensor))


def is_normal_number(x, dtype):
    """
    Check if a value is a normal number (not NaN, Inf, or subnormal).
    
    Args:
        x: torch.Tensor, numpy array, or pandas Series
        dtype: str, either "float32" or "bfloat16"
    
    Returns:
        Boolean mask (numpy array) indicating normal numbers
    """
    # Convert to numpy array
    if isinstance(x, torch.Tensor):
        x_np = x.numpy()
    elif isinstance(x, pd.Series):
        x_np = x.values
    else:
        x_np = np.asarray(x)
    
    # Check for NaN and Inf
    is_finite = np.isfinite(x_np)
    
    # Check for subnormals
    min_normal = 2**-126
    if dtype not in ["float32", "bfloat16"]:
        raise ValueError(f"Unsupported dtype: {dtype}")
    
    is_not_subnormal = np.abs(x_np) >= min_normal
    
    return is_finite & is_not_subnormal


def generate_summary(implementations, golden_unary_op, operation_name, dest_dir, dtype, impl_results_dict=None):
    """
    Generate a summary CSV file with accuracy metrics for each implementation.
    
    Args:
        implementations: List of (ttnn_op, implementation_name) tuples
        golden_unary_op: Golden reference function
        operation_name: Base operation name (e.g., "tanh", "exp")
        dest_dir: Destination directory for results
        dtype: Data type string ("float32" or "bfloat16")
        impl_results_dict: Optional dictionary mapping impl_name to list of DataFrames with detailed results
                          If None, will read from CSV files
    """
    summary_rows = []
    
    torch_dtype = getattr(torch, dtype)
    ttnn_dtype = getattr(ttnn, dtype)

    # Load detailed results if not provided
    if impl_results_dict is None:
        impl_results_dict = {}
        # Try to read from CSV files (need to find the group_size from existing files)
        import glob
        pattern = f"{dest_dir}/{operation_name}/*-{dtype}-*.csv"
        files = glob.glob(pattern)
        if files:
            # Extract group_size from first file name
            import re
            match = re.search(r'\[(\d+)\]', files[0])
            if match:
                group_size = int(match.group(1))
                for _, impl_name in implementations:
                    csv_path = f"{dest_dir}/{operation_name}/{impl_name}-{dtype}-[{group_size}].csv"
                    if os.path.exists(csv_path):
                        impl_results_dict[impl_name] = [pd.read_csv(csv_path)]
    
    # Process each implementation
    for ttnn_unary_op, implementation_name in implementations:
        try:
            # Get max ULP error (excluding special numbers and subnormals)
            max_ulp_error = np.nan
            
            if implementation_name in impl_results_dict:
                all_df = pd.concat(impl_results_dict[implementation_name])
                
                # Filter out special numbers and subnormals
                # Check if input is normal
                x_normal_mask = is_normal_number(all_df["base_x"], dtype)
                yref_normal_mask = is_normal_number(all_df["base_yref"], dtype)
                normal_mask = x_normal_mask & yref_normal_mask
                
                if np.any(normal_mask):
                    max_ulp_error = np.nanmax(all_df.loc[normal_mask, "max_ulp_error"].values)
            
            # Compute values at specific points: x=0, x=1, x=+inf, x=-inf
            test_points = {
                "x_0": torch.tensor([0.0], dtype=torch_dtype),
                "x_1": torch.tensor([1.0], dtype=torch_dtype),
                "x_pos_inf": torch.tensor([float('inf')], dtype=torch_dtype),
                "x_neg_inf": torch.tensor([float('-inf')], dtype=torch_dtype),
            }
            
            values_at_points = {}
            for point_name, test_input in test_points.items():
                try:
                    # Run TTNN operation
                    ttnn_input = ttnn.from_torch(test_input, device=device, dtype=ttnn_dtype, layout=ttnn.TILE_LAYOUT)
                    ttnn_output = ttnn.zeros_like(ttnn_input)
                    ttnn_result = ttnn_unary_op(ttnn_input, output_tensor=ttnn_output)
                    result_torch = ttnn.to_torch(ttnn_result)
                    values_at_points[point_name] = result_torch.item()
                except Exception as e:
                    logger.warning(f"Error computing {point_name} for {implementation_name}: {e}")
                    values_at_points[point_name] = np.nan
            
            summary_rows.append({
                "implementation": implementation_name,
                "max_ulp_error": max_ulp_error,
                "value_at_x_0": values_at_points["x_0"],
                "value_at_x_1": values_at_points["x_1"],
                "value_at_x_pos_inf": values_at_points["x_pos_inf"],
                "value_at_x_neg_inf": values_at_points["x_neg_inf"],
            })
        except Exception as e:
            logger.warning(f"Error generating summary for {implementation_name}: {e}")
            # Still add a row with NaN values
            summary_rows.append({
                "implementation": implementation_name,
                "max_ulp_error": np.nan,
                "value_at_x_0": np.nan,
                "value_at_x_1": np.nan,
                "value_at_x_pos_inf": np.nan,
                "value_at_x_neg_inf": np.nan,
            })
    
    # Add golden function row
    test_points = {
        "x_0": torch.tensor([0.0], dtype=torch_dtype),
        "x_1": torch.tensor([1.0], dtype=torch_dtype),
        "x_pos_inf": torch.tensor([float('inf')], dtype=torch_dtype),
        "x_neg_inf": torch.tensor([float('-inf')], dtype=torch_dtype),
    }
    
    golden_values_at_points = {}
    for point_name, test_input in test_points.items():
        try:
            # Convert to float64 for golden function
            test_input_fp64 = test_input.to(torch.float64)
            # Try with out parameter first, fallback to without
            try:
                golden_output = torch.zeros_like(test_input_fp64, dtype=torch.float64)
                golden_result = golden_unary_op(test_input_fp64, out=golden_output)
            except TypeError:
                # Function doesn't accept out parameter
                golden_result = golden_unary_op(test_input_fp64)
            # Convert back to target dtype
            golden_result_dtype = golden_result.to(torch_dtype)
            golden_values_at_points[point_name] = golden_result_dtype.item()
        except Exception as e:
            logger.warning(f"Error computing {point_name} for golden function: {e}")
            golden_values_at_points[point_name] = np.nan
    
    summary_rows.append({
        "implementation": "golden",
        "max_ulp_error": 0.0,  # Golden function has 0 ULP error by definition
        "value_at_x_0": golden_values_at_points["x_0"],
        "value_at_x_1": golden_values_at_points["x_1"],
        "value_at_x_pos_inf": golden_values_at_points["x_pos_inf"],
        "value_at_x_neg_inf": golden_values_at_points["x_neg_inf"],
    })
    
    # Create summary DataFrame and save
    summary_df = pd.DataFrame(summary_rows)
    summary_path = f"{dest_dir}/{operation_name}/summary[{dtype}].csv"
    summary_df.to_csv(summary_path, na_rep="NaN", index=False)
    print(f"Saved summary to {summary_path}")


def measure_binary_op_accuracy(implementations, golden_binary_op, operation_name, dest_dir, dtype):
    """Measure accuracy of a binary operation."""
    assert device is not None
    print(f"device =\n{device}")

    ttnn_dtype = getattr(ttnn, dtype)
    torch_dtype = getattr(torch, dtype)

    df_all_results = pd.DataFrame()
    batch_size = 128

    # Initialize result storage for each implementation
    impl_results = {impl_name: [] for _, impl_name in implementations}

    # Ensure output directory exists
    os.makedirs(f"{dest_dir}/{operation_name}", exist_ok=True)

    i = 0
    for tensor_a, tensor_b in utils.generate_binary_tensors_bf16():
        print(f"Iteration = {i}")

        # Run OP
        golden_result = golden_binary_op(tensor_a.to(torch.float32), tensor_b.to(torch.float32))
        # Ideally, we would like to only flush subnormals when plottings.
        # But this would be inconvenient here because we group take and take maximum errors.
        golden_result = flush_subnormals(golden_result)
    
        # Compute ULP resolution of golden output
        golden_ulp = ulp(golden_result.to(torch_dtype)).to(torch.float32)
        golden_result_f32 = golden_result.to(torch.float32)
        
        size = tensor_a.size()

        for ttnn_binary_op, implementation_name in implementations:

            ttnn_tensor_a = ttnn.from_torch(tensor_a, device=device, dtype=ttnn_dtype, layout=ttnn.TILE_LAYOUT)
            ttnn_tensor_b = ttnn.from_torch(tensor_b, device=device, dtype=ttnn_dtype, layout=ttnn.TILE_LAYOUT)

            ttnn_output = ttnn.zeros(size, dtype=ttnn_dtype, device=device, layout=ttnn.TILE_LAYOUT)
            ttnn_output = ttnn_binary_op(ttnn_tensor_a, ttnn_tensor_b)
            
            calculated_torch_output = ttnn.to_torch(ttnn_output).to(torch.float32)

            ulp_delta = torch.abs(calculated_torch_output - golden_result_f32) / golden_ulp

            # Reduce values to same mantissa
            # This should give 1D tensors with 2**9 elements,
            # Each elements is the min/max/mean/... error for a pair of (mantissa_a, mantissa_b)
            ulp_batch = reduce_on_batch_and_cols(ulp_delta)

            rows = 2**9

            assert ulp_batch["min"].size() == torch.Size([rows])
            assert ulp_batch["max"].size() == torch.Size([rows])
            assert ulp_batch["mean"].size() == torch.Size([rows])

            # We must reduce on batch + columns
            series_a_reduced = reduce_batch_and_cols(tensor_a.to(torch.float32))  # Note: 128 different mantissas
            series_b_reduced = reduce_batch_and_cols(
                tensor_b.to(torch.float32)
            )  # Note: tensor has 128 elements, but all have same mantissa
            assert series_a_reduced.size() == torch.Size([rows])
            assert series_b_reduced.size() == torch.Size([rows])

            # Insert into results
            df_results = pd.DataFrame({
                "a": series_a_reduced,
                "b": series_b_reduced,
                "max_ulp_error": ulp_batch["max"],
                "operation": implementation_name,
                "dtype": dtype,
            })

            impl_results[implementation_name].append(df_results)

        i += 1

    # Write results to CSV
    for _, implementation_name in implementations:
        all_df = pd.concat(impl_results[implementation_name])
        all_df.to_csv(f"{dest_dir}/{operation_name}/{implementation_name}[{dtype}].csv", na_rep="NaN", index_label="index")


def execute_benchmarks(measurement_fun, operations_dict, dest_dir, operation_name_filter=None, **kwargs):
    """
    Execute benchmarks for operations in the given operations dictionary.
    
    Args:
        measurement_fun: Function to call for measuring accuracy (e.g., measure_op_accuracy_bf16, measure_op_accuracy_f32)
        operations_dict: Dictionary of operations (e.g., UNARY_OPERATIONS, BINARY_OPERATIONS)
        dest_dir: Destination directory for results
        operation_name_filter: Optional operation name to filter by (if None, process all operations)
        group_size: Optional group size parameter to pass to measurement function
        **kwargs: Additional keyword arguments to pass to measurement function
    
    Returns:
        Tuple of (successfull_operations, failed_operations)
    """
    success_count = 0
    successfull_operations = []
    failed_operations = []

    # Calculate total group count (operations that match the filter)
    total_group_cnt = 0
    if operation_name_filter is not None:
        total_group_cnt = 1 if operation_name_filter in operations_dict else 0
    else:
        total_group_cnt = len(operations_dict)
    group_cnt = 0
    
    for operation_name, op_data in operations_dict.items():
        if operation_name_filter is not None and operation_name != operation_name_filter:
            continue

        golden_op = get_golden_function(operations_dict, operation_name)
        implementations = [(ttnn_op_impl, impl_name) for impl_name, ttnn_op_impl in op_data["implementations"].items()]

        group_cnt += 1
        impl_count = len(implementations)
        print(f"\nProcessing group {group_cnt}/{total_group_cnt}: {operation_name} ({impl_count} implementation(s))")
        
        try:
            start_time = time.time()
            
            # Call measurement function with implementations, golden_op, operation_name, dest_dir, and optional group_size
            
            measurement_fun(implementations, golden_op, operation_name, dest_dir, **kwargs)

            end_time = time.time()
            elapsed_s = end_time - start_time
            print(f"Group {operation_name} completed in {elapsed_s:.4f}s")

            success_count += impl_count
            successfull_operations.extend([impl_name for _, impl_name in implementations])
        except Exception as e:
            logger.warning(f"Could not run operation group {operation_name}: {e}")
            logger.warning(f"{traceback.format_exc()}")
            failed_operations.extend([impl_name for _, impl_name in implementations])

    # Calculate total implementation count (accounting for filter)
    if operation_name_filter is not None:
        total_impl_count = len(operations_dict[operation_name_filter]["implementations"]) if operation_name_filter in operations_dict else 0
    else:
        total_impl_count = sum(len(op_data['implementations']) for op_data in operations_dict.values())
    
    print(f"\nSucessfully ran {success_count} / {total_impl_count} operations")
    print(f"{TERM_GREEN}SUCCESS: {successfull_operations}{TERM_RESET}")
    logger.warning(f"FAILED: {failed_operations}")
    
    return (successfull_operations, failed_operations)



def main(args, operation_type=None):
    from arg_parser import create_parser, validate_operation
    
    # Create a parser to get operation name first (using unary parser which has all args)
    basic_parser = create_parser("unary")  # Use unary as template (has all args including group-size)
    
    # Parse to get operation name first (without type-specific validation)
    # args is sys.argv, so skip the script name
    temp_args = basic_parser.parse_args(args[1:])
    
    # If operation is specified, auto-detect the type
    if temp_args.operation is not None:
        operation_type = detect_operation_type(temp_args.operation)
        # Validate the operation with the detected type
        temp_args.operation, temp_args.type = validate_operation(temp_args.operation, operation_type, temp_args.type)
    else:
        # No operation specified, use provided operation_type or default to "unary"
        if operation_type is None:
            operation_type = "unary"
    
    # Use the parsed args (they're already validated if operation was specified)
    # Note: group-size will be None for binary operations, which is fine
    args = temp_args

    # Set numpy floating point warning to reduce stdout clutter
    # Since we test *all* possible floating point values, invalid values
    # are expected.
    np.seterr(divide="ignore")
    np.seterr(invalid="ignore")
    np.seterr(over="ignore")

    if operation_type == "binary":
        # Binary operations use a different measurement approach
        all_operations = iterate_all_operations(BINARY_OPERATIONS)
        dest_dir = f"accuracy_results/results/binary/"
        os.makedirs(dest_dir, exist_ok=True)

        # Use execute_benchmarks to process all unary operations
        (successfull_operations, failed_operations) = execute_benchmarks(
            measurement_fun=measure_binary_op_accuracy,
            operations_dict=BINARY_OPERATIONS,
            dest_dir=dest_dir,
            operation_name_filter=args.operation,
            dtype=args.type,
        )
    else:
        # Unary operations
        if args.group_size is None:
            if args.type == "bfloat16":
                group_size = 1
            elif args.type == "float32":
                group_size = 65536
            else:
                raise ValueError(f"Invalid data type: {args.type}")
        else:
            group_size = args.group_size

        dest_dir = f"accuracy_results/results/unary/"
        os.makedirs(dest_dir, exist_ok=True)

        # Select measurement function based on data type
        if args.type == "bfloat16":
            measurement_fun = measure_op_accuracy_bf16
        elif args.type == "float32":
            measurement_fun = measure_op_accuracy_f32
        else:
            raise ValueError(f"Invalid data type: {args.type}")

        # Use execute_benchmarks to process all unary operations
        (successfull_operations, failed_operations) = execute_benchmarks(
            measurement_fun=measurement_fun,
            operations_dict=UNARY_OPERATIONS,
            dest_dir=dest_dir,
            operation_name_filter=args.operation,
            group_size=group_size
        )


if __name__ == "__main__":
    args = sys.argv
    main(args, operation_type="unary")
    ttnn.close_device(device)
