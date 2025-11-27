
sfpi_inline sfpi::vFloat _sfpu_round_(sfpi::vFloat z, sfpi::vInt& k_int) {
    const sfpi::vFloat HALF = sfpi::vFloat(0.5f);

    sfpi::vFloat abs_z = sfpi::abs(z);

    sfpi::vFloat z_plus_half = abs_z + sfpi::vFloat(HALF);
    k_int = sfpu::_float_to_int32_(z_plus_half);  // Truncation of z+0.5 ≈ rounding of z
    sfpi::vFloat k = sfpi::int32_to_float(k_int, 0);

    v_if (z < -0.5f) {
        k_int = -k_int;
        k = -k;
    }
    v_endif;

    return k;
}

/*
 * This function implements ln(x) using polynomial approximation.
 * Algorithm based on ln_fp32 from operations.py:
 * 1. Handle special cases (x <= 0, infinity, NaN)
 * 2. Extract exponent and mantissa: x = 2^n × m
 * 3. Reduce range: adjust m to be in [sqrt(2)/2, sqrt(2)]
 * 4. Compute ln(m) using polynomial approximation
 * 5. Return n×ln(2) + ln(m)
 *
 * @param val The input value (sfpi::vFloat vector), can be any floating point number
 * @return sfpi::vFloat Result of ln(val)
 */
template <bool is_fp32_dest_acc_en = false>
sfpi_inline sfpi::vFloat calculate_sfpi_kernel(sfpi::vFloat val) {
    sfpi::vFloat result = sfpi::vConst0;
    
    // Constants (precomputed in fp32)
    constexpr float LN2 = 0.6931471805599453f;      // ln(2)
    constexpr float SQRT2 = 1.4142135623730951f;   // sqrt(2)
    constexpr float HALF = 0.5f;
    constexpr float ONE = 1.0f;
    constexpr float TWO = 2.0f;
    constexpr float ZERO = 0.0f;
    constexpr int BIAS = 127;
    
    // Polynomial coefficients for ln(m) where m in [sqrt(2)/2, sqrt(2)]
    // ln(m) = 2z(1 + z²/3 + z⁴/5 + z⁶/7 + z⁸/9 + z¹⁰/11)
    // where z = (m - 1) / (m + 1)
    constexpr float c11 = 0.09090909090909091f;  // 1/11
    constexpr float c9 = 0.1111111111111111f;    // 1/9
    constexpr float c7 = 0.14285714285714285f;   // 1/7
    constexpr float c5 = 0.2f;                    // 1/5
    constexpr float c3 = 0.3333333333333333f;    // 1/3
    
    // Check for special cases
    sfpi::vInt exp_bits = sfpi::exexp(val); // extract biased exponent
    sfpi::vInt man_bits = sfpi::exman9(val);
    sfpi::vInt signbit = sfpi::reinterpret<sfpi::vInt>(val) & 0x80000000;  // returns 0 for +ve value
    
    v_if(val == sfpi::vFloat(ZERO)) {
        // ln(0) = -inf
        result = -std::numeric_limits<float>::infinity();
    }
    v_elseif(val < sfpi::vFloat(ZERO)) {
        // ln(negative) = NaN
        result = std::numeric_limits<float>::quiet_NaN();
    }
    v_elseif(exp_bits == 255 && man_bits != 0) {
        // NaN: exponent = 255 (all 1s) and mantissa != 0
        result = std::numeric_limits<float>::quiet_NaN();
    }
    v_elseif(signbit == 0 && exp_bits == 255 && man_bits == 0) {
        // +infinity
        result = std::numeric_limits<float>::infinity();
    }
    v_else {
        
        
        // Extract exponent (without bias) BEFORE normalizing
        // exexp() already returns the debiased exponent (unlike exexp_nodebias())
        // We need to extract it before we modify the value with setexp
        sfpi::vInt exp = sfpi::exexp(val);
        
        // Extract mantissa and construct m in [1, 2)
        // Use setexp to normalize to [1, 2) range by setting exponent to 127 (bias)
        sfpi::vFloat m = sfpi::setexp(val, 127);
        
        // Range reduction: if m >= sqrt(2), divide by 2 and increment exponent
        // All operations in fp32
        v_if(m >= sfpi::vFloat(SQRT2)) {
            m = m * sfpi::vFloat(HALF);
            exp = exp + 1;
        }
        v_endif;
        
        // Now m is in [sqrt(2)/2, sqrt(2)] ≈ [0.707, 1.414]
        // Transform to z = (m - 1) / (m + 1) for better convergence
        // This maps m ∈ [0.707, 1.414] to z ∈ [-0.172, 0.172]
        // ln(m) = 2 × (z + z³/3 + z⁵/5 + z⁷/7 + ...)
        
        // ALL operations happen on fp32 values
        sfpi::vFloat m_minus_1 = m - sfpi::vFloat(ONE);
        sfpi::vFloat m_plus_1 = m + sfpi::vFloat(ONE);

        sfpi::vFloat inv_m_plus_1 = ckernel::sfpu::_sfpu_reciprocal_<2>(m_plus_1);
        sfpi::vFloat z = m_minus_1 * inv_m_plus_1;
        sfpi::vFloat z2 = z * z;
        
        // Polynomial approximation using odd powers
        // ln(m) = 2z(1 + z²/3 + z⁴/5 + z⁶/7 + z⁸/9 + z¹⁰/11)
        // Using Horner's method with ALL operations in fp32
        sfpi::vFloat p = sfpi::vFloat(c11);
        
        sfpi::vFloat temp = z2 * p;
        p = sfpi::vFloat(c9) + temp;
        
        temp = z2 * p;
        p = sfpi::vFloat(c7) + temp;
        
        temp = z2 * p;
        p = sfpi::vFloat(c5) + temp;
        
        temp = z2 * p;
        p = sfpi::vFloat(c3) + temp;
        
        temp = z2 * p;
        p = sfpi::vFloat(ONE) + temp;
        
        // Final computation: ln(m) = 2 * z * p
        temp = z * p;
        sfpi::vFloat ln_m = sfpi::vFloat(TWO) * temp;
        
        // Combine: ln(x) = exp×ln(2) + ln(m)
        // Convert exp to float and multiply by ln(2)
        // Note: exp can be negative for x < 0.5, so we need to handle sign correctly
        // int32_to_float should handle negative values, but ensure proper conversion
        sfpi::vFloat expf = sfpi::int32_to_float(exp, 0);
        temp = expf * sfpi::vFloat(LN2);
        result = temp + ln_m;
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
    ckernel::sfpu::_init_reciprocal_<false, false>();
}

