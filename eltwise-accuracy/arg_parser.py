import argparse
import sys
from operations import UNARY_OPERATIONS, BINARY_OPERATIONS


def get_available_operations(operation_type="unary"):
    """Get list of available operations for the specified type."""
    if operation_type == "unary":
        return list(UNARY_OPERATIONS.keys())
    elif operation_type == "binary":
        return list(BINARY_OPERATIONS.keys())
    else:
        raise ValueError("operation_type must be 'unary' or 'binary'")


def validate_operation(operation_name, operation_type="unary", operation_dtype="bfloat16"):
    """Validate that the operation exists and return available operations if not."""
    available_ops = get_available_operations(operation_type)
    
    if operation_name not in available_ops:
        print(f"Error: Operation '{operation_name}' not found in {operation_type} operations.")
        print(f"Available {operation_type} operations:")
        for op in sorted(available_ops):
            print(f"  {op}")
        sys.exit(1)
    

    available_types = ["float32", "bfloat16"]
    if operation_dtype not in available_types:
        print(f"Error: Operation '{operation_dtype}' not found in available types.")
        print(f"Available types:")
        for op in sorted(available_types):
            print(f"  {op}")
        sys.exit(1)

    return operation_name, operation_dtype


def create_parser(operation_type="unary"):
    """Create argument parser for the specified operation type."""
    parser = argparse.ArgumentParser(
        description=f"Measure accuracy of TTNN {operation_type} operations"
    )
    
    parser.add_argument(
        "--operation", "-o",
        type=str,
        required=False,
        help=f"Name of the {operation_type} operation to test. If not provided, runs all operations from config file."
    )
    
    parser.add_argument(
        "--output-dir", "-O",
        type=str,
        default="accuracy_results/results/",
        help="Output directory for results (default: accuracy_results/results/)"
    )
    
    parser.add_argument(
        "--type", "-t",
        type=str,
        default="bfloat16",
        help="Operation data type (default: bfloat16). Must be one of: float32, bfloat16"
    )

    if operation_type == "unary":
        parser.add_argument(
            "--group-size", "-g",
            type=int,
            default=1,
            help="Group size for bfloat16 measurements (default: 32)"
        )

    return parser


def parse_args(operation_type="unary"):
    """Parse command line arguments and validate operation."""
    parser = create_parser(operation_type)
    args = parser.parse_args()
    
    # Validate the operation exists only if provided
    if args.operation:
        (args.operation, args.type) = validate_operation(args.operation, operation_type, args.type)
    
    return args
