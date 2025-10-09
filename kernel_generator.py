import os
import ttnn
import torch
import math

from jinja2 import Environment, FileSystemLoader



def serialize_polynomial_coeff_to_horner(polynomial_coefficients):
    if not polynomial_coefficients:
        return "0.0"
    
    if len(polynomial_coefficients) == 1:
        return str(polynomial_coefficients[0])
    
    # Start with the highest degree coefficient
    result = str(polynomial_coefficients[0])
    
    # Work backwards through the coefficients
    for coeff in polynomial_coefficients[1:]:
        result = f"{coeff} + frac * ({result})"
    
    return result

def generate_sfpi_kernel_source_code_with_polynomial(jinja_env, sfpi_kernel_name, polynomial_coefficients):

    template = jinja_env.get_template(f"sfpi/{sfpi_kernel_name}.cpp.j2")

    polynomial_expression = serialize_polynomial_coeff_to_horner(polynomial_coefficients)

    kernel_source_code = template.render(
        SFPI_POLY_APPROX=polynomial_expression,
    )

    return kernel_source_code
    

def generate_kernel_source_code_from_polynomial(kernel_name, sfpi_kernel_name, polynomial_coefficients):

    kernel_name = "unary"

    jinja_env = Environment(
        loader=FileSystemLoader("kernel_templates"),
    )

    sfpu_kernel_code = generate_sfpi_kernel_source_code_with_polynomial(jinja_env, sfpi_kernel_name, polynomial_coefficients)
    
    # For debugging purposes
    with open(f"sfpu_kernel_{sfpi_kernel_name}.cpp", "w") as f:
        f.write(sfpu_kernel_code)

    template = jinja_env.get_template(f"{kernel_name}.cpp.j2")

    kernel_source_code = template.render(
        SFPU_KERNEL_NAME="calculate_sfpi_kernel",
        SFPU_KERNEL_IMPL=sfpu_kernel_code,
    )

    return kernel_source_code

def base_unary_kernel(compute_kernel_source_code, ttnn_input_tensor, device, metal_home_dir):


    output_tensor = ttnn.zeros_like(ttnn_input_tensor)

    io_tensors = [ttnn_input_tensor, output_tensor]

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
    writer_compile_time_args.extend(ttnn.TensorAccessorArgs(output_tensor).get_compile_time_args())
    compute_compile_time_args = [num_tiles, 1]
    reader_rt_args = [ttnn_input_tensor.buffer_address(), num_tiles, 0]
    writer_rt_args = [output_tensor.buffer_address(), num_tiles, 0]

    reader_kernel_descriptor = ttnn.KernelDescriptor(
        kernel_source=f"{metal_home_dir}/ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/reader_unary_interleaved_start_id.cpp",
        source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
        core_ranges=core_grid,
        compile_time_args=reader_compile_time_args,
        runtime_args=[[reader_rt_args]],
        config=ttnn.ReaderConfigDescriptor(),
    )
    writer_kernel_descriptor = ttnn.KernelDescriptor(
        kernel_source=f"{metal_home_dir}/ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/writer_unary_interleaved_start_id.cpp",
        source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
        core_ranges=core_grid,
        compile_time_args=writer_compile_time_args,
        runtime_args=[[writer_rt_args]],
        config=ttnn.WriterConfigDescriptor(),
    )

    sfpu_defines = []
    # struct ComputeConfigDescriptor {
    # using UnpackToDestModes = std::vector<UnpackToDestMode>;

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
        # kernel_source=f"{metal_home_dir}/tt_metal/kernels/compute/eltwise_sfpu.cpp",
        kernel_source=compute_kernel_source_code,
        # source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH, expecting this to be the default value
        source_type=ttnn.KernelDescriptor.SourceType.SOURCE_CODE,
        core_ranges=core_grid,
        compile_time_args=compute_compile_time_args,
        defines=sfpu_defines,
        runtime_args=[[[]]],
        config=compute_kernel_config,
    )


    

    program_descriptor = ttnn.ProgramDescriptor(
        kernels=[reader_kernel_descriptor, writer_kernel_descriptor, compute_kernel_descriptor],
        semaphores=[],
        cbs=[in_cb_descriptor, out_cb_descriptor],
    )

    output = ttnn.generic_op(io_tensors, program_descriptor)

    print(f"fp32_dest_acc_en = {compute_kernel_config.fp32_dest_acc_en}")
    print(f"dtype = {ttnn_input_tensor.dtype}")
    print(f"output.dtype = {output.dtype}")  

    return output



def generate_unary_kernel_from_polynomial(sfpi_kernel_name, polynomial_coefficients, full_name=None):

    TT_METAL_HOME = os.getenv("TT_METAL_HOME")

    compute_kernel_source_code = generate_kernel_source_code_from_polynomial("unary", sfpi_kernel_name, polynomial_coefficients)
    
    # For debugging purposes
    if full_name is None:
        full_name = sfpi_kernel_name

    with open(f"compute_unary_{full_name}.cpp", "w") as f:
        f.write(compute_kernel_source_code)

    # Doudble lambda to properly copy value of kernel_source_code
    fun =  lambda tensor, device, kernel_source_code=compute_kernel_source_code: \
                    (base_unary_kernel(kernel_source_code, tensor, device, TT_METAL_HOME))
    
    return fun


def test_generated_kernel(function):

    device = ttnn.open_device(device_id=0)

    shape = [256, 256]
    value = 8.03125
    tensor = torch.full(shape, value, dtype=torch.float32)

    ttnn_tensor = ttnn.from_torch(tensor, device=device, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT)

    output_tensor = function(ttnn_tensor, device)

    torch_calculated_output = ttnn.to_torch(output_tensor)

    print(f"torch_calculated_output =\n{torch_calculated_output}")

    ttnn.close_device(device)
    

def main():

    function = generate_unary_kernel_from_polynomial("exp", [0.34228965640068054,0.652752697467804,1.0022648572921753])

    test_generated_kernel(function)

if __name__ == "__main__":
    main()