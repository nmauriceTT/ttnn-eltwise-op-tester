
template <bool is_fp32_acc_to_dest_mode = true>
sfpi_inline sfpi::vFloat calculate_sfpi_kernel(sfpi::vFloat val) {

    // log(1/x) = -log(x)
    // log2(x) = exexp(x) + log2(setexp(x, 127))

    // Normalize base to calculation range       // set base as positive
    sfpi::vFloat x = sfpi::setexp(val, 127);

    // 3rd order polynomial approx - determined using rminimax over [1,2]
    sfpi::vFloat series_result = x * (x * (x * 0x2.44734p-4f - 0xd.e712ap-4f) + 0x2.4f5388p+0f) - 0x1.952992p+0f;

    // Convert exponent to float
    sfpi::vInt exp = sfpi::exexp(val);
    v_if(exp < 0) { exp = sfpi::setsgn(~exp + 1, 1); }
    v_endif;
    sfpi::vFloat exp_f32 = sfpi::int32_to_float(exp, 0);

    sfpi::vFloat vConst1Ln2 = 1.4426950408889634f;
    sfpi::vFloat result = exp_f32 + series_result * vConst1Ln2;  // exp correction: ln(1+x) + exp*ln(2)

    if constexpr(!is_fp32_acc_to_dest_mode) {
        result = sfpi::reinterpret<sfpi::vFloat>(sfpi::float_to_fp16b(result, 0));
    }

    return result;
}

inline void calculate_sfpi_kernel_init()
{
}