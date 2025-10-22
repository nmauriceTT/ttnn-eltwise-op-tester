
template <bool is_fp32_acc_to_dest_mode = true>
sfpi_inline sfpi::vFloat calculate_sfpi_kernel(sfpi::vFloat x) {

    sfpi::vFloat val = sfpi::setsgn(x, 0); // set positive

    sfpi::vFloat result = val * (0.999004364013671875 + val * (3.0897438526153564453125e-2 + val * (-0.4890659749507904052734375 + val * (0.281917631626129150390625 + val * (-6.6649019718170166015625e-2 + val * (5.876733921468257904052734375e-3))))));

    sfpi::vFloat threshold_value = sfpi::vFloat(1.0f);
    sfpi::vec_min_max(result, threshold_value);

    result = sfpi::setsgn(result, x); // restore sign (i.e. tanh(-x) = -tanh(x))

    if constexpr (!is_fp32_acc_to_dest_mode) {
        result = sfpi::reinterpret<sfpi::vFloat>(sfpi::float_to_fp16b(result, 0));
    }

    return result;
}

inline void calculate_sfpi_kernel_init()
{
}