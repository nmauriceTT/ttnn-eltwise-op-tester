template <bool is_fp32_acc_to_dest_mode = true>
sfpi_inline sfpi::vFloat calculate_sfpi_kernel(sfpi::vFloat val) {

    // https://varietyofsound.wordpress.com/2011/02/14/efficient-tanh-computation-using-lamberts-continued-fraction/

    sfpi::vFloat x = sfpi::setsgn(val, 0); // set positive

    sfpi::vFloat x2 = x * x;
    sfpi::vFloat numerator = x * (135135.f + x2 * (17326.f + x2 * (378.f + x2)));
    sfpi::vFloat denominator = 135135.f + x2 * (62370.f + x2 * (3150.f + 28.f * x2));

    sfpi::vFloat result = ckernel::sfpu::_sfpu_reciprocal_<2>(denominator);
    result = result * numerator;
    
    sfpi::vFloat threshold_value = sfpi::vFloat(1.0f);
    sfpi::vec_min_max(result, threshold_value);

    result = sfpi::setsgn(result, val); // restore sign (i.e. tanh(-x) = -tanh(x))

    if constexpr (!is_fp32_acc_to_dest_mode) {
        result = sfpi::reinterpret<sfpi::vFloat>(sfpi::float_to_fp16b(result, 0));
    }

    return result;
}

inline void calculate_sfpi_kernel_init()
{
    ckernel::sfpu::_init_reciprocal_<false, false>();
}