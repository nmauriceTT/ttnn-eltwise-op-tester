"""Run a single TTNN eltwise op once to compile its kernel (no tracy)."""

import sys
import traceback

import ttnn

from src.operations import BINARY_OPERATIONS, UNARY_OPERATIONS, get_operation_variant_by_name


def _shard_shape(dtype, operation_type):
    if operation_type == "binary":
        shard_shape = [256, 128]
        if dtype == "float32":
            shard_shape = [128, 128]
        return shard_shape

    shard_shape = [256, 256]
    if dtype == "float32":
        shard_shape = [256, 128]
    return shard_shape


def main():
    args = sys.argv[1:]
    if len(args) != 3:
        print(f"Error: invalid arguments {args}. Expected: <unary|binary> <dtype> <implementation_name>")
        sys.exit(1)

    operation_type, dtype, implementation_name = args
    if operation_type not in ("unary", "binary"):
        print(f"Error: operation type must be unary or binary, got {operation_type!r}")
        sys.exit(1)

    operations = UNARY_OPERATIONS if operation_type == "unary" else BINARY_OPERATIONS
    ttnn_operation, _ = get_operation_variant_by_name(operations, implementation_name)

    device = None
    try:
        device = ttnn.open_device(device_id=0)
        grid = device.core_grid
        shard_shape = _shard_shape(dtype, operation_type)
        shape = (1, 1, grid.y * grid.x * shard_shape[0], shard_shape[1])
        mem_config = ttnn.create_sharded_memory_config(
            shape,
            core_grid=grid,
            strategy=ttnn.ShardStrategy.HEIGHT,
            orientation=ttnn.ShardOrientation.ROW_MAJOR,
        )
        ttnn_dtype = getattr(ttnn, dtype)

        if operation_type == "binary":
            input_a = ttnn.full(
                shape=shape, fill_value=2.0, dtype=ttnn_dtype, device=device,
                layout=ttnn.TILE_LAYOUT, memory_config=mem_config,
            )
            input_b = ttnn.full(
                shape=shape, fill_value=3.0, dtype=ttnn_dtype, device=device,
                layout=ttnn.TILE_LAYOUT, memory_config=mem_config,
            )
            _ = ttnn_operation(input_a, input_b)
        else:
            input_tensor = ttnn.full(
                shape=shape, fill_value=1.0, dtype=ttnn_dtype, device=device,
                layout=ttnn.TILE_LAYOUT, memory_config=mem_config,
            )
            output_tensor = ttnn.zeros_like(input_tensor)
            _ = ttnn_operation(input_tensor, output_tensor=output_tensor)

        ttnn.synchronize_device(device)
        ttnn.close_device(device)
    except Exception as e:
        print(f"Error running {implementation_name}: {e}", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        if device is not None:
            try:
                ttnn.close_device(device)
            except Exception:
                pass
        sys.exit(1)


if __name__ == "__main__":
    main()
