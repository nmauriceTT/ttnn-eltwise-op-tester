
sfpi_inline sfpi::vFloat _sfpu_round_(sfpi::vFloat z, sfpi::vInt& k_int) {
    const sfpi::vFloat HALF = sfpi::vFloat(0.5f);

    sfpi::vFloat abs_z = sfpi::abs(z);

    sfpi::vFloat z_plus_half = abs_z + sfpi::vFloat(HALF);
    k_int = sfpu::_float_to_int32_(z_plus_half);  // Truncation of z+0.5 ≈ rounding of z
    sfpi::vFloat k = sfpi::int32_to_float(k_int, 0);

    v_if (z < -0.5f) {
        k_int = -k_int;
        k = -k;
        // k_int = -k_int + 1;
        // k = sfpi::int32_to_float(k_int, 0);
    }
    v_endif;

    return k;
}

/*
 * This function implements exp(x) using polynomial approximation.
 * Algorithm based on exp_fp32 from operations.py:
 * 1. Handle special cases (overflow, underflow, NaN)
 * 2. Convert to base-2: exp(x) = 2^(x/ln2)
 * 3. Range reduction: z = x/ln2 = k + r where k is integer, r in [-0.5, 0.5]
 * 4. Compute 2^r using polynomial approximation of exp(r*ln2)
 * 5. Scale by 2^k: result = 2^k * 2^r
 *
 * @param val The input value (sfpi::vFloat vector), can be any floating point number
 * @return sfpi::vFloat Result of exp(val)
 */
template <bool is_fp32_dest_acc_en = false>
sfpi_inline sfpi::vFloat calculate_sfpi_kernel(sfpi::vFloat val) {
    sfpi::vFloat result = sfpi::vConst0;
    
    // Constants (all in fp32)
    constexpr float INV_LN2 = 1.4426950408889634f;  // 1/ln(2)
    constexpr float LN2 = 0.6931471805599453f;      // ln(2)
    constexpr float OVERFLOW_THRESHOLD = 88.0f;
    constexpr float UNDERFLOW_THRESHOLD = -103.97f;
    constexpr float HALF = 0.5f;
    
    // Polynomial coefficients for exp(s) where s in [-0.5*ln(2), 0.5*ln(2)] ≈ [-0.3466, 0.3466]
    // exp(s) = 1 + s + s²/2! + s³/3! + s⁴/4! + s⁵/5! + s⁶/6!
    constexpr float c6 = 1.0f / 720.0f;  // 1/6!
    constexpr float c5 = 1.0f / 120.0f;  // 1/5!
    constexpr float c4 = 1.0f / 24.0f;   // 1/4!
    constexpr float c3 = 1.0f / 6.0f;    // 1/3!
    constexpr float c2 = 0.5f;           // 1/2!
    constexpr float c1 = 1.0f;           // 1/1!
    constexpr float c0 = 1.0f;           // constant
    
    // Clamp to prevent overflow/underflow
    sfpi::vFloat threshold_high = sfpi::vFloat(OVERFLOW_THRESHOLD);
    sfpi::vFloat threshold_low = sfpi::vFloat(UNDERFLOW_THRESHOLD);
    vec_min_max(threshold_low, val);
    vec_min_max(val, threshold_high);
    
    // Check for special cases
    sfpi::vInt exp_bits = sfpi::exexp(val);
    sfpi::vInt man_bits = sfpi::exman9(val);
    
    v_if(exp_bits == 255 && man_bits != 0) {
        // NaN: exponent = 255 (all 1s) and mantissa != 0
        result = std::numeric_limits<float>::quiet_NaN();
    }
    v_elseif(val >= sfpi::vFloat(OVERFLOW_THRESHOLD)) {
        // Overflow
        result = std::numeric_limits<float>::infinity();
    }
    v_elseif(val <= sfpi::vFloat(UNDERFLOW_THRESHOLD)) {
        // Underflow
        result = sfpi::vConst0;
    }
    v_else {
        // Step 1: Convert to base-2 exponential
        // z = x / ln(2) = x * (1/ln(2))
        sfpi::vFloat z = val * sfpi::vFloat(INV_LN2);
        
        // Step 2: Range reduction
        // Split z = k + r where k is integer, r in [-0.5, 0.5]
        // k = floor(z + 0.5) - rounding to nearest integer
        // sfpi::vFloat z_plus_half = z + sfpi::vFloat(HALF);
        // sfpi::vInt k_int = sfpu::_float_to_int32_(z_plus_half);  // Truncation of z+0.5 ≈ rounding of z
        // sfpi::vFloat k = sfpi::int32_to_float(k_int, 0);
        
        sfpi::vInt k_int;
        sfpi::vFloat k = _sfpu_round_(z, k_int);

        // Compute r = z - k (r is in [-0.5, 0.5])
        sfpi::vFloat r = z - k;
        
        // Step 3: Polynomial approximation for 2^r = exp(r*ln2)
        // Compute s = r * ln(2), all in fp32
        // s is in [-0.5*ln(2), 0.5*ln(2)] ≈ [-0.3466, 0.3466]
        sfpi::vFloat s = r * sfpi::vFloat(LN2);
        
        // Approximate exp(s) using Taylor series via Horner's method
        // exp(s) = 1 + s + s²/2! + s³/3! + s⁴/4! + s⁵/5! + s⁶/6!
        // p = c0 + s*(c1 + s*(c2 + s*(c3 + s*(c4 + s*(c5 + s*c6)))))
        sfpi::vFloat p = sfpi::vFloat(c6);
        
        sfpi::vFloat temp = p * s;
        p = sfpi::vFloat(c5) + temp;
        
        temp = p * s;
        p = sfpi::vFloat(c4) + temp;
        
        temp = p * s;
        p = sfpi::vFloat(c3) + temp;
        
        temp = p * s;
        p = sfpi::vFloat(c2) + temp;
        
        temp = p * s;
        p = sfpi::vFloat(c1) + temp;
        
        temp = p * s;
        p = sfpi::vFloat(c0) + temp;
        
        // Step 4: Scale by 2^k using exponent manipulation
        // ldexp(p, k_int) = p * 2^k
        // We do this by adding k_int to the exponent of p
        // Get the current exponent of p (without bias)
        sfpi::vInt p_exp = sfpi::exexp_nodebias(p);
        // Add k_int to get the new exponent
        sfpi::vInt new_exp = p_exp + k_int;
        
        // Set the new exponent (setexp expects exponent without bias, similar to exp_21f)
        result = sfpi::reinterpret<sfpi::vFloat>(
            sfpi::setexp(p, new_exp));
    }
    v_endif;
    
    if constexpr (!is_fp32_dest_acc_en) {
        // LRegs work on float32 data. If DST is bfloat16 then SFPSTORE will truncate it.
        // This can reduce accuracy: for instance, 9**2 = 80.8 gets round to 80.5
        // rather than 81 (which would have been correct).
        // To avoid this issue, we explicitly convert to bfloat16 using round-to-nearest-even.
        result = sfpi::reinterpret<sfpi::vFloat>(sfpi::float_to_fp16b(result, 0));
    }
    
    return result;
}

void calculate_sfpi_kernel_init() {
}