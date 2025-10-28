
template <bool is_fp32_acc_to_dest_mode = true>
sfpi_inline sfpi::vFloat calculate_sfpi_kernel(sfpi::vFloat x) {

    // sfpi::vFloat value = sfpi::setsgn(x, 1); // x = -x
    sfpi::vFloat value = -x;
    value = ckernel::sfpu::_sfpu_exp_21f_<true>(value);
    value = sfpi::vConst1 + value;

    sfpi::vFloat result;
    if constexpr(is_fp32_acc_to_dest_mode) {
        result = ckernel::sfpu::_sfpu_reciprocal_<2>(value);
    } else {
        result = ckernel::sfpu::_sfpu_reciprocal_<1>(value);
        result = sfpi::reinterpret<sfpi::vFloat>(sfpi::float_to_fp16b(result, 0));
    }

    return result;
}

inline void calculate_sfpi_kernel_init()
{
    ckernel::sfpu::_init_reciprocal_<false, false>();
}