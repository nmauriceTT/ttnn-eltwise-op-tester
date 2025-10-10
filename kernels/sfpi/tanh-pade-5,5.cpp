template <bool is_fp32_acc_to_dest_mode = true>
sfpi_inline sfpi::vFloat calculate_sfpi_kernel(sfpi::vFloat x) {

    // Pade approximation of tanh

    // Clip input to [-4, 4]
    sfpi::vFloat threshold_high = sfpi::vFloat(4.f);
    sfpi::vFloat threshold_low = sfpi::vFloat(-4.f);
    sfpi::vec_min_max(threshold_low, x);
    sfpi::vec_min_max(x, threshold_high);

    sfpi::vFloat x2 = x * x;
    sfpi::vFloat x3 = x2 * x;
    sfpi::vFloat x4 = x2 * x2;
    sfpi::vFloat x5 = x3 * x2;

    sfpi::vFloat numerator = (x + x3 * sfpi::vFloat(1.f/9.f) + x5 * sfpi::vFloat(1.f/945.f));
    sfpi::vFloat denominator = (sfpi::vFloat(1.f) + sfpi::vFloat(4.f/9.f) * x2 + x4 * sfpi::vFloat(1.f/63.f));
    
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