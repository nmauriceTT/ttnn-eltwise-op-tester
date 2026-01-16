import os
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
    compute_kernel_config.unpack_to_dest_mode = [ttnn._ttnn.program_descriptor.UnpackToDestMode.UnpackToDestFp32] * 32 #  ttnn.UnpackToDestMode.UnpackToDestFp32
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
