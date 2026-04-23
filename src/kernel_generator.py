import os
import struct
import ttnn
import torch
import math

from jinja2 import Environment, FileSystemLoader



def serialize_polynomial_coeff(polynomial_coefficients):
    if not polynomial_coefficients:
        return "0.0"

    return ','.join([str(coeff) for coeff in polynomial_coefficients])

def generate_sfpi_kernel_source_code_with_polynomial(jinja_env, sfpi_kernel_name, polynomial_coefficients):

    template = jinja_env.get_template(f"sfpi/{sfpi_kernel_name}.cpp.j2")

    polynomial_coefficients = serialize_polynomial_coeff(polynomial_coefficients)

    kernel_source_code = template.render(
        POLY_COEFFS=polynomial_coefficients,
    )

    return kernel_source_code
    

def generate_kernel_from_sfpi_source(kernel_name, sfpi_kernel_name):

    kernel_name = "unary-sfpi"

    jinja_env = Environment(
        loader=FileSystemLoader("kernel_templates"),
    )

    sfpi_kernel_code = ""
    with open(f"kernels/sfpi/{sfpi_kernel_name}.cpp", "r") as f:
        sfpi_kernel_code = f.read()
    
    template = jinja_env.get_template(f"{kernel_name}.cpp.j2")

    kernel_source_code = template.render(
        SFPU_KERNEL_NAME=f"calculate_sfpi_kernel",
        SFPU_KERNEL_IMPL=sfpi_kernel_code,
    )

    return kernel_source_code

def generate_kernel_from_tti_source(tti_kernel_name):
    kernel_name = "unary-tti"

    jinja_env = Environment(
        loader=FileSystemLoader("kernel_templates"),
    )

    with open(f"kernels/tti/{tti_kernel_name}.cpp", "r") as f:
        tti_kernel_code = f.read()

    template = jinja_env.get_template(f"unary-tti.cpp.j2")

    kernel_source_code = template.render(
        SFPU_KERNEL_NAME=f"calculate_tti_kernel",
        SFPU_KERNEL_IMPL=tti_kernel_code,
    )

    return kernel_source_code

def generate_kernel_source_code_from_polynomial(kernel_name, sfpi_kernel_name, polynomial_coefficients):

    kernel_name = "unary-sfpi"

    jinja_env = Environment(
        loader=FileSystemLoader("kernel_templates"),
    )

    sfpu_kernel_code = generate_sfpi_kernel_source_code_with_polynomial(jinja_env, sfpi_kernel_name, polynomial_coefficients)
    
    # For debugging purposes
    with open(f"sfpu_kernel_{sfpi_kernel_name}.cpp", "w") as f:
        f.write(sfpu_kernel_code)

    template = jinja_env.get_template(f"{kernel_name}.cpp.j2")

    kernel_source_code = template.render(
        SFPU_KERNEL_NAME=f"calculate_sfpi_kernel",
        SFPU_KERNEL_IMPL=sfpu_kernel_code,
    )

    return kernel_source_code

def generate_kernel_source_code_from_llk(kernel_name, llk_init, llk_name):

    kernel_name = "unary"
    jinja_env = Environment(
        loader=FileSystemLoader("kernel_templates"),
    )

    template = jinja_env.get_template(f"{kernel_name}.cpp.j2")

    kernel_source_code = template.render(
        SFPU_LLK_INIT_NAME=llk_init,
        SFPU_LLK_NAME=llk_name,
    )
    return kernel_source_code


def generate_binary_kernel_source_code_from_llk(llk_init, llk_op):
    """Render binary.cpp.j2 with the given LLK init and per-tile call names."""
    jinja_env = Environment(
        loader=FileSystemLoader("kernel_templates"),
    )
    template = jinja_env.get_template("binary.cpp.j2")
    return template.render(LLK_OP_INIT=llk_init, LLK_OP=llk_op)


def generic_binary_kernel(
    compute_kernel_source_code,
    ttnn_a,
    ttnn_b,
    ttnn_output=None,
    core_grid=None,
    metal_home_dir=None,
):
    """
    Run a binary compute kernel with HiFi4 + fp32_dest_acc_en=True.

    CBs:  c_0=in0, c_1=in1, c_2=out
    Reader:  reader_binary_interleaved_start_id.cpp
    Writer:  writer_unary_interleaved_start_id.cpp
    Compute: caller-supplied source (rendered from binary.cpp.j2)
    """
    if metal_home_dir is None:
        metal_home_dir = os.getenv("TT_METAL_HOME")
        if metal_home_dir is None:
            raise RuntimeError("TT_METAL_HOME environment variable is not set")

    if ttnn_output is None:
        ttnn_output = ttnn.zeros_like(ttnn_a)

    if core_grid is None:
        core = ttnn.CoreCoord(0, 0)
        core_grid = ttnn.CoreRangeSet([ttnn.CoreRange(core, core)])
    elif isinstance(core_grid, ttnn.CoreGrid):
        grid_coord = ttnn.CoreCoord(core_grid.x - 1, core_grid.y - 1)
        core_grid = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), grid_coord)])

    dtype = ttnn_a.dtype
    bytes_per_datum = 4 if dtype == ttnn.float32 else 2
    cb_page_size = bytes_per_datum * 1024
    cb_total_size = 2 * cb_page_size

    def _make_cb(buf_index):
        fmt = ttnn.CBFormatDescriptor(
            buffer_index=buf_index,
            data_format=dtype,
            page_size=cb_page_size,
        )
        return ttnn.CBDescriptor(
            total_size=cb_total_size,
            core_ranges=core_grid,
            format_descriptors=[fmt],
        )

    num_tiles = math.prod(ttnn_a.shape) // (32 * 32)

    # Reader: reader_binary_interleaved_start_id.cpp
    # Compile-time: [block_or_width_sharded, ...TensorAccessorArgs_a..., ...TensorAccessorArgs_b...]
    reader_ct_args = [0]  # block_or_width_sharded = False
    reader_ct_args.extend(ttnn.TensorAccessorArgs(ttnn_a).get_compile_time_args())
    reader_ct_args.extend(ttnn.TensorAccessorArgs(ttnn_b).get_compile_time_args())

    # Runtime: src0_addr, src1_addr, num_tiles, start_id, block_h, block_w, num_cores_y
    reader_rt_vals = [ttnn_a.buffer_address(), ttnn_b.buffer_address(), num_tiles, 0, 1, num_tiles, 1]

    # Writer: writer_unary_interleaved_start_id.cpp
    # Compile-time: [out_cb_index, ...TensorAccessorArgs_out...]
    writer_ct_args = [2]  # out CB is c_2 (index 2)
    writer_ct_args.extend(ttnn.TensorAccessorArgs(ttnn_output).get_compile_time_args())

    # Runtime: dst_addr, num_tiles, start_id
    writer_rt_vals = [ttnn_output.buffer_address(), num_tiles, 0]

    reader_rt_args = ttnn.RuntimeArgs()
    writer_rt_args = ttnn.RuntimeArgs()
    for core_range in core_grid.ranges():
        for x in range(core_range.start.x, core_range.end.x + 1):
            for y in range(core_range.start.y, core_range.end.y + 1):
                reader_rt_args[x][y] = reader_rt_vals
                writer_rt_args[x][y] = writer_rt_vals

    reader_config = ttnn.DataMovementConfigDescriptor(
        processor=ttnn.DataMovementProcessor.RISCV_1,
        noc=ttnn.NOC.RISCV_1_default,
    )
    writer_config = ttnn.DataMovementConfigDescriptor(
        processor=ttnn.DataMovementProcessor.RISCV_0,
        noc=ttnn.NOC.RISCV_0_default,
    )

    reader_kernel = ttnn.KernelDescriptor(
        kernel_source=(
            f"{metal_home_dir}/ttnn/cpp/ttnn/operations/eltwise/binary"
            "/device/kernels/dataflow/reader_binary_interleaved_start_id.cpp"
        ),
        source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
        core_ranges=core_grid,
        compile_time_args=reader_ct_args,
        runtime_args=reader_rt_args,
        config=reader_config,
    )
    writer_kernel = ttnn.KernelDescriptor(
        kernel_source=(
            f"{metal_home_dir}/ttnn/cpp/ttnn/operations/eltwise/unary"
            "/device/kernels/dataflow/writer_unary_interleaved_start_id.cpp"
        ),
        source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
        core_ranges=core_grid,
        compile_time_args=writer_ct_args,
        runtime_args=writer_rt_args,
        config=writer_config,
    )

    compute_config = ttnn.ComputeConfigDescriptor()
    compute_config.fp32_dest_acc_en = True
    compute_config.math_approx_mode = False
    compute_config.math_fidelity = ttnn.MathFidelity.HiFi4

    compute_kernel = ttnn.KernelDescriptor(
        kernel_source=compute_kernel_source_code,
        source_type=ttnn.KernelDescriptor.SourceType.SOURCE_CODE,
        core_ranges=core_grid,
        compile_time_args=[num_tiles],
        defines=[],
        runtime_args=[],
        config=compute_config,
    )

    program = ttnn.ProgramDescriptor(
        kernels=[reader_kernel, writer_kernel, compute_kernel],
        semaphores=[],
        cbs=[_make_cb(0), _make_cb(1), _make_cb(2)],
    )

    return ttnn.generic_op([ttnn_a, ttnn_b, ttnn_output], program)


def make_generic_binary_kernel_op(llk_init, llk_op):
    """
    Return a callable (ttnn_a, ttnn_b) -> ttnn_output suitable for use in
    BINARY_OPERATIONS.  The kernel is compiled with HiFi4 + fp32_dest_acc_en=True.

    Example:
        multiply_fp32acc = make_generic_binary_kernel_op("mul_tiles_init", "mul_tiles")
    """
    source = generate_binary_kernel_source_code_from_llk(llk_init, llk_op)
    return lambda a, b: generic_binary_kernel(source, a, b)


def _f32_to_u32(value: float) -> int:
    """Return the IEEE 754 single-precision bit pattern of *value* as a uint32."""
    return struct.unpack("I", struct.pack("f", value))[0]


def generic_binary_kernel_with_dst_init(
    compute_kernel_source_code,
    ttnn_a,
    ttnn_b,
    dst_init_f32: float,
    ttnn_output=None,
    core_grid=None,
    metal_home_dir=None,
):
    """
    Run the binary_with_dst_init kernel: for every tile, Dst = C + A * B.

    *dst_init_f32* is the float32 constant C encoded as a compile-time arg.
    All other arguments mirror generic_binary_kernel().
    """
    if metal_home_dir is None:
        metal_home_dir = os.getenv("TT_METAL_HOME")
        if metal_home_dir is None:
            raise RuntimeError("TT_METAL_HOME environment variable is not set")

    if ttnn_output is None:
        ttnn_output = ttnn.zeros_like(ttnn_a)

    if core_grid is None:
        core = ttnn.CoreCoord(0, 0)
        core_grid = ttnn.CoreRangeSet([ttnn.CoreRange(core, core)])
    elif isinstance(core_grid, ttnn.CoreGrid):
        grid_coord = ttnn.CoreCoord(core_grid.x - 1, core_grid.y - 1)
        core_grid = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), grid_coord)])

    dtype = ttnn_a.dtype
    bytes_per_datum = 4 if dtype == ttnn.float32 else 2
    cb_page_size = bytes_per_datum * 1024
    cb_total_size = 2 * cb_page_size

    def _make_cb(buf_index):
        fmt = ttnn.CBFormatDescriptor(
            buffer_index=buf_index,
            data_format=dtype,
            page_size=cb_page_size,
        )
        return ttnn.CBDescriptor(
            total_size=cb_total_size,
            core_ranges=core_grid,
            format_descriptors=[fmt],
        )

    num_tiles = math.prod(ttnn_a.shape) // (32 * 32)
    dst_init_u32 = _f32_to_u32(dst_init_f32)

    reader_ct_args = [0]
    reader_ct_args.extend(ttnn.TensorAccessorArgs(ttnn_a).get_compile_time_args())
    reader_ct_args.extend(ttnn.TensorAccessorArgs(ttnn_b).get_compile_time_args())

    reader_rt_vals = [ttnn_a.buffer_address(), ttnn_b.buffer_address(), num_tiles, 0, 1, num_tiles, 1]

    writer_ct_args = [2]
    writer_ct_args.extend(ttnn.TensorAccessorArgs(ttnn_output).get_compile_time_args())
    writer_rt_vals = [ttnn_output.buffer_address(), num_tiles, 0]

    reader_rt_args = ttnn.RuntimeArgs()
    writer_rt_args = ttnn.RuntimeArgs()
    for core_range in core_grid.ranges():
        for x in range(core_range.start.x, core_range.end.x + 1):
            for y in range(core_range.start.y, core_range.end.y + 1):
                reader_rt_args[x][y] = reader_rt_vals
                writer_rt_args[x][y] = writer_rt_vals

    reader_config = ttnn.DataMovementConfigDescriptor(
        processor=ttnn.DataMovementProcessor.RISCV_1,
        noc=ttnn.NOC.RISCV_1_default,
    )
    writer_config = ttnn.DataMovementConfigDescriptor(
        processor=ttnn.DataMovementProcessor.RISCV_0,
        noc=ttnn.NOC.RISCV_0_default,
    )

    reader_kernel = ttnn.KernelDescriptor(
        kernel_source=(
            f"{metal_home_dir}/ttnn/cpp/ttnn/operations/eltwise/binary"
            "/device/kernels/dataflow/reader_binary_interleaved_start_id.cpp"
        ),
        source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
        core_ranges=core_grid,
        compile_time_args=reader_ct_args,
        runtime_args=reader_rt_args,
        config=reader_config,
    )
    writer_kernel = ttnn.KernelDescriptor(
        kernel_source=(
            f"{metal_home_dir}/ttnn/cpp/ttnn/operations/eltwise/unary"
            "/device/kernels/dataflow/writer_unary_interleaved_start_id.cpp"
        ),
        source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
        core_ranges=core_grid,
        compile_time_args=writer_ct_args,
        runtime_args=writer_rt_args,
        config=writer_config,
    )

    compute_config = ttnn.ComputeConfigDescriptor()
    compute_config.fp32_dest_acc_en = True
    compute_config.math_approx_mode = False
    compute_config.math_fidelity = ttnn.MathFidelity.HiFi4

    compute_kernel = ttnn.KernelDescriptor(
        kernel_source=compute_kernel_source_code,
        source_type=ttnn.KernelDescriptor.SourceType.SOURCE_CODE,
        core_ranges=core_grid,
        compile_time_args=[num_tiles, dst_init_u32],
        defines=[],
        runtime_args=[],
        config=compute_config,
    )

    program = ttnn.ProgramDescriptor(
        kernels=[reader_kernel, writer_kernel, compute_kernel],
        semaphores=[],
        cbs=[_make_cb(0), _make_cb(1), _make_cb(2)],
    )

    return ttnn.generic_op([ttnn_a, ttnn_b, ttnn_output], program)

def generate_kernel_from_source_path(source_path):

    source_code = None
    with open(source_path, "r") as f:
        source_code = f.read()
    
    return source_code




# ---------------------------------------------------------------------------
# Ternary kernel:  output = Dst_init + A * B
# Uses copy_tile (loads Dst_init into Dst[0]) + accumulating ELWMUL
# (binary_tiles_init with acc_to_dest=true, so Dst is not zeroed before MUL)
# ---------------------------------------------------------------------------


def generic_ternary_mul_accum_kernel(
    kernel_source_code,
    ttnn_a,
    ttnn_b,
    ttnn_dst_init,
    ttnn_output=None,
    core_grid=None,
    metal_home_dir=None,
):
    """
    Run the ternary accumulating multiply kernel:
        output[tile] = Dst_init[tile] + A[tile] * B[tile]

    The kernel pre-loads Dst_init via copy_tile, then runs ELWMUL with
    acc_to_dest=true (no ZEROACC), giving the exact single-instruction
    Dst += A * B behaviour.

    CBs:  c_0=A, c_1=B, c_2=Dst_init  →  c_3=output
    Compiler config: HiFi4, fp32_dest_acc_en=True
    """
    if metal_home_dir is None:
        metal_home_dir = os.getenv("TT_METAL_HOME")
        if metal_home_dir is None:
            raise RuntimeError("TT_METAL_HOME environment variable is not set")

    if ttnn_output is None:
        ttnn_output = ttnn.zeros_like(ttnn_a)

    if core_grid is None:
        core = ttnn.CoreCoord(0, 0)
        core_grid = ttnn.CoreRangeSet([ttnn.CoreRange(core, core)])
    elif isinstance(core_grid, ttnn.CoreGrid):
        grid_coord = ttnn.CoreCoord(core_grid.x - 1, core_grid.y - 1)
        core_grid = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), grid_coord)])

    dtype = ttnn_a.dtype
    bytes_per_datum = 4 if dtype == ttnn.float32 else 2
    cb_page_size = bytes_per_datum * 1024
    cb_total_size = 2 * cb_page_size

    def _make_cb(buf_index, cb_dtype=None):
        data_format = cb_dtype if cb_dtype is not None else dtype
        bytes_per_datum = 4 if data_format == ttnn.float32 else 2
        page_size = bytes_per_datum * 1024

        fmt = ttnn.CBFormatDescriptor(
            buffer_index=buf_index,
            data_format=data_format,
            page_size=page_size,
        )
        return ttnn.CBDescriptor(
            total_size=2 * page_size,
            core_ranges=core_grid,
            format_descriptors=[fmt],
        )

    num_tiles = math.prod(ttnn_a.shape) // (32 * 32)

    # Reader: ternary_reader_nobcast_ttt.cpp from tt-metal ternary ops.
    # CT args layout (as expected by that file):
    #   [cb_id_in0=0, cb_id_in1=1, cb_id_in2=2,
    #    ...TensorAccessorArgs(A)...,  (offset 3)
    #    ...TensorAccessorArgs(B)...,
    #    ...TensorAccessorArgs(Dst_init)...]
    # RT args: [src0_addr, src1_addr, src2_addr, num_tiles, start_id]
    reader_ct_args = [0, 1, 2]  # CB IDs for the three inputs
    reader_ct_args.extend(ttnn.TensorAccessorArgs(ttnn_a).get_compile_time_args())
    reader_ct_args.extend(ttnn.TensorAccessorArgs(ttnn_b).get_compile_time_args())
    reader_ct_args.extend(ttnn.TensorAccessorArgs(ttnn_dst_init).get_compile_time_args())

    reader_rt_vals = [
        ttnn_a.buffer_address(),
        ttnn_b.buffer_address(),
        ttnn_dst_init.buffer_address(),
        num_tiles,
        0,  # start_id
    ]

    # Writer: existing unary writer, output CB is c_3
    writer_ct_args = [3]  # out CB index
    writer_ct_args.extend(ttnn.TensorAccessorArgs(ttnn_output).get_compile_time_args())
    writer_rt_vals = [ttnn_output.buffer_address(), num_tiles, 0]

    reader_rt_args = ttnn.RuntimeArgs()
    writer_rt_args = ttnn.RuntimeArgs()
    for core_range in core_grid.ranges():
        for x in range(core_range.start.x, core_range.end.x + 1):
            for y in range(core_range.start.y, core_range.end.y + 1):
                reader_rt_args[x][y] = reader_rt_vals
                writer_rt_args[x][y] = writer_rt_vals

    reader_config = ttnn.DataMovementConfigDescriptor(
        processor=ttnn.DataMovementProcessor.RISCV_1,
        noc=ttnn.NOC.RISCV_1_default,
    )
    writer_config = ttnn.DataMovementConfigDescriptor(
        processor=ttnn.DataMovementProcessor.RISCV_0,
        noc=ttnn.NOC.RISCV_0_default,
    )

    reader_kernel = ttnn.KernelDescriptor(
        kernel_source=(
            f"{metal_home_dir}/ttnn/cpp/ttnn/operations/eltwise/ternary"
            "/device/kernels/dataflow/ternary_reader_nobcast_ttt.cpp"
        ),
        source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
        core_ranges=core_grid,
        compile_time_args=reader_ct_args,
        runtime_args=reader_rt_args,
        config=reader_config,
    )
    writer_kernel = ttnn.KernelDescriptor(
        kernel_source=(
            f"{metal_home_dir}/ttnn/cpp/ttnn/operations/eltwise/unary"
            "/device/kernels/dataflow/writer_unary_interleaved_start_id.cpp"
        ),
        source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
        core_ranges=core_grid,
        compile_time_args=writer_ct_args,
        runtime_args=writer_rt_args,
        config=writer_config,
    )

    compute_config = ttnn.ComputeConfigDescriptor()
    compute_config.fp32_dest_acc_en = True
    compute_config.math_approx_mode = False
    compute_config.math_fidelity = ttnn.MathFidelity.HiFi4

    unpack_to_dest_mode = [
        ttnn._ttnn.program_descriptor.UnpackToDestMode.Default] * 32 #  ttnn.UnpackToDestMode.UnpackToDestFp32
    unpack_to_dest_mode[2] = ttnn._ttnn.program_descriptor.UnpackToDestMode.UnpackToDestFp32
    compute_config.unpack_to_dest_mode = unpack_to_dest_mode

    compute_kernel = ttnn.KernelDescriptor(
        kernel_source=kernel_source_code,
        source_type=ttnn.KernelDescriptor.SourceType.SOURCE_CODE,
        core_ranges=core_grid,
        compile_time_args=[num_tiles],
        defines=[],
        runtime_args=[],
        config=compute_config,
    )

    program = ttnn.ProgramDescriptor(
        kernels=[reader_kernel, writer_kernel, compute_kernel],
        semaphores=[],
        # CB c_2 (Dst_init) uses float32 so the full 32-bit DST value is
        # preserved when copy_tile loads it into the FP32 destination register.
        # CBs c_0, c_1, c_3 keep the dtype of the A/B/output tensors (BF16).
        cbs=[_make_cb(0), _make_cb(1), _make_cb(2, ttnn.float32), _make_cb(3)],
    )

    return ttnn.generic_op([ttnn_a, ttnn_b, ttnn_dst_init, ttnn_output], program)


def generic_unary_kernel(compute_kernel_source_code, ttnn_input_tensor, ttnn_output_tensor=None, core_grid=None, metal_home_dir=None):

    if isinstance(core_grid, ttnn.CoreGrid):
        grid_coord = ttnn.CoreCoord(core_grid.x - 1, core_grid.y - 1)
        core_grid = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), grid_coord)])

    if metal_home_dir is None:
        metal_home_dir = os.getenv("TT_METAL_HOME")
        if metal_home_dir is None:
            raise RuntimeError("TT_METAL_HOME environment variable is not set")

    assert ttnn_output_tensor is not None

    io_tensors = [ttnn_input_tensor, ttnn_output_tensor]

    if core_grid is None:
        core = ttnn.CoreCoord(0, 0)
        core_grid = ttnn.CoreRangeSet([ttnn.CoreRange(core, core)])

    input_cb_data_format = ttnn_input_tensor.dtype  # this will be mapped tt::DataFormat::Float16_b

    if input_cb_data_format == ttnn.float32:
        bytes_per_datum = 4
    else:
        bytes_per_datum = 2


    cb_total_size = 2 * bytes_per_datum * 1024  # tt::DataFormat::Float16_b hard coded to have size 2 * 1024
    cb_page_size = bytes_per_datum * 1024

    in_cb = 0
    out_cb = 1
    in_cb_format = ttnn.CBFormatDescriptor(
        buffer_index=in_cb,
        data_format=input_cb_data_format,
        page_size=cb_page_size,
    )
    out_cb_format = ttnn.CBFormatDescriptor(
        buffer_index=out_cb,
        data_format=input_cb_data_format,
        page_size=cb_page_size,
    )
    in_cb_descriptor = ttnn.CBDescriptor(
        total_size=cb_total_size,
        core_ranges=core_grid,
        format_descriptors=[in_cb_format],
    )
    out_cb_descriptor = ttnn.CBDescriptor(
        total_size=cb_total_size,
        core_ranges=core_grid,
        format_descriptors=[out_cb_format],
    )

    tile_shape = ttnn_input_tensor.get_tile().tile_shape
    tile_volume = tile_shape[0] * tile_shape[1]
    tensor_volume = math.prod(ttnn_input_tensor.shape)
    num_tiles = tensor_volume // tile_volume

    reader_compile_time_args = ttnn.TensorAccessorArgs(ttnn_input_tensor).get_compile_time_args()
    writer_compile_time_args = [out_cb]
    writer_compile_time_args.extend(ttnn.TensorAccessorArgs(ttnn_output_tensor).get_compile_time_args())
    compute_compile_time_args = [num_tiles, 1]

    shared_reader_rt_args = [ttnn_input_tensor.buffer_address(), num_tiles, 0]
    shared_writer_rt_args = [ttnn_output_tensor.buffer_address(), num_tiles, 0]

    reader_rt_args = ttnn.RuntimeArgs()
    writer_rt_args = ttnn.RuntimeArgs()
    for core_range in core_grid.ranges():
        for x in range(core_range.start.x, core_range.end.x + 1):
            for y in range(core_range.start.y, core_range.end.y + 1):
                reader_rt_args[x][y] = shared_reader_rt_args
                writer_rt_args[x][y] = shared_writer_rt_args

    # Reader runs on NCRISC (RISCV_1), Writer runs on BRISC (RISCV_0)
    reader_config = ttnn.DataMovementConfigDescriptor(
        processor=ttnn.DataMovementProcessor.RISCV_1,
        noc=ttnn.NOC.RISCV_1_default,
    )
    writer_config = ttnn.DataMovementConfigDescriptor(
        processor=ttnn.DataMovementProcessor.RISCV_0,
        noc=ttnn.NOC.RISCV_0_default,
    )

    reader_kernel_descriptor = ttnn.KernelDescriptor(
        kernel_source=f"{metal_home_dir}/ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/reader_unary_interleaved_start_id.cpp",
        source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
        core_ranges=core_grid,
        compile_time_args=reader_compile_time_args,
        runtime_args=reader_rt_args,
        config=reader_config,
    )
    writer_kernel_descriptor = ttnn.KernelDescriptor(
        kernel_source=f"{metal_home_dir}/ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/writer_unary_interleaved_start_id.cpp",
        source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
        core_ranges=core_grid,
        compile_time_args=writer_compile_time_args,
        runtime_args=writer_rt_args,
        config=writer_config,
    )

    sfpu_defines = []
    # MathFidelity math_fidelity = MathFidelity::HiFi4;
    # bool fp32_dest_acc_en = false;
    # bool dst_full_sync_en = false;
    # UnpackToDestModes unpack_to_dest_mode;
    # bool bfp8_pack_precise = false;
    # bool math_approx_mode = false;
    # };

    compute_kernel_config = ttnn.ComputeConfigDescriptor()
    # compute_kernel_config.dst_full_sync_en = True
    compute_kernel_config.fp32_dest_acc_en = ttnn_input_tensor.dtype == ttnn.float32
    compute_kernel_config.math_approx_mode = False
    compute_kernel_config.math_fidelity = ttnn.MathFidelity.HiFi4
    # compute_kernel_config.unpack_to_dest_mode = [ttnn._ttnn.program_descriptor.UnpackToDestMode.UnpackToDestFp32] * 32 #  ttnn.UnpackToDestMode.UnpackToDestFp32
    # math_fidelity=ttnn.MathFidelity.HiFi4,

    compute_kernel_descriptor = ttnn.KernelDescriptor(
        kernel_source=compute_kernel_source_code,
        source_type=ttnn.KernelDescriptor.SourceType.SOURCE_CODE,
        core_ranges=core_grid,
        compile_time_args=compute_compile_time_args,
        defines=sfpu_defines,
        runtime_args=[],
        config=compute_kernel_config,
    )

    program_descriptor = ttnn.ProgramDescriptor(
        kernels=[reader_kernel_descriptor, writer_kernel_descriptor, compute_kernel_descriptor],
        semaphores=[],
        cbs=[in_cb_descriptor, out_cb_descriptor],
    )

    output = ttnn.generic_op(io_tensors, program_descriptor)

    return output


def generate_unary_kernel_from_sfpi_source(sfpi_kernel_name):
    return generate_kernel_from_sfpi_source("unary-sfpi", sfpi_kernel_name)


def generate_unary_kernel_from_polynomial(sfpi_kernel_name, polynomial_coefficients, full_name=None, core_grid=None):

    compute_kernel_source_code = generate_kernel_source_code_from_polynomial("unary-sfpi", sfpi_kernel_name, polynomial_coefficients)
    
    # For debugging purposes
    if full_name is None:
        full_name = sfpi_kernel_name

    os.makedirs("generated", exist_ok=True)

    with open(f"generated/compute_unary_{full_name}.cpp", "w") as f:
        f.write(compute_kernel_source_code)

    return compute_kernel_source_code
