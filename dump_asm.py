import argparse
import os
import sys

from src import operations
from src.asm_dump import dump_implementation_asm


def list_available_unary_operations():
    return [
        (variant_name, base_operation_name)
        for variant_name, base_operation_name, _, _ in operations.iterate_all_operations(operations.UNARY_OPERATIONS)
    ]


def list_available_binary_operations():
    return [
        (variant_name, base_operation_name)
        for variant_name, base_operation_name, _, _ in operations.iterate_all_operations(operations.BINARY_OPERATIONS)
    ]


def detect_operation_type(operation_name):
    if operation_name in operations.UNARY_OPERATIONS:
        return "unary"
    if operation_name in operations.BINARY_OPERATIONS:
        return "binary"
    return None


def main(args):
    allowed_dtypes = ["float32", "bfloat16", "uint16", "uint32"]

    parser = argparse.ArgumentParser(
        description="Disassemble TTNN eltwise kernel MATH-thread (trisc1) ELF and print SFPU instruction histograms"
    )
    parser.add_argument(
        "--operation", "-k",
        type=str,
        default=None,
        help="Filter by base operation name (runs all variants). Default: all operations.",
    )
    parser.add_argument(
        "--dtype", "-t",
        type=str,
        default="bfloat16",
        help=f"Data type (default: bfloat16). Must be one of: {', '.join(allowed_dtypes)}",
    )
    parser.add_argument(
        "--type",
        type=str,
        default="unary",
        choices=["unary", "binary"],
        help="Operation type (default: unary).",
    )
    parser.add_argument(
        "--output-dir", "-o",
        type=str,
        default=None,
        help="Directory for .asm files (default: generated/asm/<type>/)",
    )

    parsed_args = parser.parse_args(args)

    operation_type = parsed_args.type
    if parsed_args.operation:
        detected_type = detect_operation_type(parsed_args.operation)
        if detected_type is None:
            unary_ops = list(operations.UNARY_OPERATIONS.keys())
            binary_ops = list(operations.BINARY_OPERATIONS.keys())
            print(f"Error: Operation '{parsed_args.operation}' not found.")
            print(f"Available unary operations: {', '.join(sorted(unary_ops))}")
            print(f"Available binary operations: {', '.join(sorted(binary_ops))}")
            sys.exit(1)
        operation_type = detected_type

    if operation_type == "binary":
        all_operations = list_available_binary_operations()
    else:
        all_operations = list_available_unary_operations()

    if parsed_args.dtype not in allowed_dtypes:
        print(f"Error: Data type '{parsed_args.dtype}' not in allowed data types.")
        print(f"Allowed data types: {', '.join(allowed_dtypes)}")
        sys.exit(1)

    if parsed_args.operation:
        all_operations = [
            (variant_name, base_operation_name)
            for variant_name, base_operation_name in all_operations
            if base_operation_name == parsed_args.operation
        ]
        if not all_operations:
            print(f"Error: Operation '{parsed_args.operation}' not found in {operation_type} operations.")
            sys.exit(1)
        print(f"Dumping asm for operation '{parsed_args.operation}' ({len(all_operations)} variants)")

    asm_out_dir = parsed_args.output_dir or f"generated/asm/{operation_type}/"
    os.makedirs(asm_out_dir, exist_ok=True)

    if not os.getenv("TT_METAL_HOME"):
        print("Error: TT_METAL_HOME is not set.")
        sys.exit(1)

    failed = []
    for implementation_name, base_operation_name in all_operations:
        print(f"Dumping asm for {base_operation_name} / {implementation_name}")
        result = dump_implementation_asm(
            implementation_name,
            base_operation_name,
            parsed_args.dtype,
            asm_out_dir,
            operation_type=operation_type,
        )
        if result is None:
            failed.append(implementation_name)

    if failed:
        print(f"Failed to dump asm for: {', '.join(failed)}")
        sys.exit(1)


if __name__ == "__main__":
    main(sys.argv[1:])
