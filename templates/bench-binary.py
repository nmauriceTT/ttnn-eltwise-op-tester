import ttnn
import sys
import traceback
from operations import BINARY_OPERATIONS, get_operation_variant_by_name


args = sys.argv[1:]
if len(args) >= 3:
    ITERATIONS = int(args[0])
    DTYPE = args[1]
    IMPLEMENTATION_NAME = args[2]
else:
    print(f"Error: Invalid arguments: {args}. Expected: <iterations> <dtype> <implementation_name>")
    sys.exit(1)

device = None
try:
    device = ttnn.open_device(device_id=0)

    shard_shape = [256, 128]
    if DTYPE == "float32":
        shard_shape = [128, 128]

    grid = ttnn.CoreGrid(x=8, y=8)

    shape = (1, 1, grid.y * grid.x * shard_shape[0], shard_shape[1])
    mem_config = ttnn.create_sharded_memory_config(
        shape,
        core_grid=grid,
        strategy=ttnn.ShardStrategy.HEIGHT,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
    )

    ttnn_dtype = getattr(ttnn, DTYPE)

    # Create two input tensors for binary operations
    # input_tensor_a = ttnn.rand(shape, dtype=ttnn_dtype, device=device, layout=ttnn.TILE_LAYOUT, memory_config=mem_config)
    # input_tensor_b = ttnn.rand(shape, dtype=ttnn_dtype, device=device, layout=ttnn.TILE_LAYOUT, memory_config=mem_config)
    input_tensor_a = ttnn.full(shape=shape, fill_value=2.0, dtype=ttnn_dtype, device=device, layout=ttnn.TILE_LAYOUT, memory_config=mem_config)
    input_tensor_b = ttnn.full(shape=shape, fill_value=1.5, dtype=ttnn_dtype, device=device, layout=ttnn.TILE_LAYOUT, memory_config=mem_config)
    output_tensor = ttnn.zeros_like(input_tensor_a)

    ttnn_operation, _ = get_operation_variant_by_name(BINARY_OPERATIONS, IMPLEMENTATION_NAME)
    print(f"Running benchmark for {IMPLEMENTATION_NAME}")
    print(f"ttnn operation: {ttnn_operation}")

    assert ITERATIONS > 0

    from tracy import signpost
    signpost(f"BENCHMARK START")
    for i in range(ITERATIONS):
        signpost(f"ITERATION START")
        _ = ttnn_operation(input_tensor_a, input_tensor_b)
        signpost(f"ITERATION END")

        # ttnn.deallocate(y)
    signpost("BENCHMARK END")

    ttnn.synchronize_device(device)
    ttnn.close_device(device)
    
except Exception as e:
    print(f"Error running benchmark for {IMPLEMENTATION_NAME}: {e}", file=sys.stderr)
    traceback.print_exc(file=sys.stderr)
    if device is not None:
        try:
            ttnn.close_device(device)
        except:
            pass
    sys.exit(1)

