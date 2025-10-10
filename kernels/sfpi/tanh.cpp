template <bool is_fp32_acc_to_dest_mode = true>
sfpi_inline sfpi::vFloat calculate_sfpi_kernel(sfpi::vFloat input) {

    sfpi::vFloat two_val = input * 2.0f;

    sfpi::vFloat exp_2x_val = ckernel::sfpu::_sfpu_exp_21f_<true>(two_val);
    
    sfpi::vFloat numerator = exp_2x_val - 1.0f;
    sfpi::vFloat denominator = exp_2x_val + 1.0f;
    
    sfpi::vFloat result = ckernel::sfpu::_sfpu_reciprocal_<2>(denominator);
    result = result * numerator;

    if constexpr (is_fp32_acc_to_dest_mode) {
        result = sfpi::reinterpret<sfpi::vFloat>(sfpi::float_to_fp16b(result, 0));
    }

    return result;
}

inline void calculate_sfpi_kernel_init()
{
    ckernel::sfpu::_init_reciprocal_<false, false>();
}