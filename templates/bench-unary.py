import ttnn
import sys
from operations import UNARY_OPERATIONS, get_operation_variant_by_name


args = sys.argv[1:]
if len(args) >= 3:
    ITERATIONS = int(args[0])
    DTYPE = args[1]
    IMPLEMENTATION_NAME = args[2]
else:
    print(f"Error: Invalid arguments: {args}. Expected: <iterations> <dtype> <implementation_name>")
    sys.exit(1)

device = ttnn.open_device(device_id=0)

shard_shape = [256, 128]
grid = ttnn.CoreGrid(x=8, y=8)

shape = (1, 1, grid.y * grid.x * shard_shape[0], shard_shape[1])
mem_config = ttnn.create_sharded_memory_config(
    shape,
    core_grid=grid,
    strategy=ttnn.ShardStrategy.HEIGHT,
    orientation=ttnn.ShardOrientation.ROW_MAJOR,
)

ttnn_dtype = getattr(ttnn, DTYPE)

# input_tensor = ttnn.rand(shape, dtype=ttnn_dtype, device=device, layout=ttnn.TILE_LAYOUT, memory_config=mem_config)
input_tensor = ttnn.full(shape=shape, fill_value=1.0, dtype=ttnn_dtype, device=device, layout=ttnn.TILE_LAYOUT, memory_config=mem_config)
output_tensor = ttnn.zeros_like(input_tensor)

ttnn_operation, _ = get_operation_variant_by_name(UNARY_OPERATIONS, IMPLEMENTATION_NAME )
print(f"Running benchmark for {IMPLEMENTATION_NAME}")
print(f"ttnn operation: {ttnn_operation}")

from tracy import signpost
signpost(f"BENCHMARK START")
for i in range(ITERATIONS):
    _ = ttnn_operation(input_tensor, output_tensor=output_tensor)
    # ttnn.deallocate(y)
signpost("BENCHMARK END")

ttnn.synchronize_device(device)
ttnn.close_device(device)