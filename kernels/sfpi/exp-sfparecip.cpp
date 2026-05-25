


template <bool is_fp32_acc_to_dest_mode>
sfpi_inline sfpi::vFloat calculate_sfpi_kernel(sfpi::vFloat val) {

    constexpr float ONE_LN2 = 1.4426950216293334961f;
    sfpi::vFloat xlog2 = (val * ONE_LN2 + 127.f);

    // Intermedirary values can overflow in xlog2 is outside of [0, 256[ which leads to invalid resutls instead of 0
    // (when input < -88.5) and +inf (when input > 88.5)
    // To avoid this, we clamp xlog2 to [0, 255]
    // (thresholds values are rounded to bf16, as it does not change result but only requires one SFPLOADI vs. two)
    sfpi::vFloat threshold_low = 0.f;
    sfpi::vFloat threshold_high = sfpi::vFloat(255.f);
    vec_min_max(threshold_low, xlog2);
    vec_min_max(xlog2, threshold_high);


    sfpi::vInt z = _float_to_int32_for_exp21f_(xlog2);

    sfpi::vInt exponential_part =
        sfpi::exexp(sfpi::reinterpret<sfpi::vFloat>(z), sfpi::ExponentMode::NoDebias);  // Extract exponent ( = 2**(integer part of val/ln2))
    sfpi::vInt fractional_part =
        sfpi::exman9(sfpi::reinterpret<sfpi::vFloat>(z));  // Extract mantissa ( = leftover part, in [0; 1])

    constexpr float TWO_POW_MINUS_23 = 1.1920928955078125e-07f;
    constexpr float LN2 = 0.6931471805599453f;
    // constexpr float factor = LN2 * TWO_POW_MINUS_23;
    constexpr float factor = 8.24220478534698486328125e-8f;

    sfpi::vFloat frac = sfpi::int32_to_float(fractional_part, sfpi::RoundMode::NearestEven);
    

    frac = sfpi::approx_exp(frac * factor);

    sfpi::vFloat y = sfpi::setexp(frac, exponential_part);

    if constexpr(!is_fp32_acc_to_dest_mode) {
        y = sfpi::convert<sfpi::vFloat16b>(y, sfpi::RoundMode::NearestEven);
    }

    return y;
}

template <bool is_fp32_acc_to_dest_mode>
inline void calculate_sfpi_kernel_init()
{
}