#define ARCH_BLACKHOLE 1

#ifdef ARCH_WORMHOLE
template <bool is_fp32_dest_acc_en, int ITERATIONS>
void calculate_tti_kernel() {

    constexpr uint32_t input_type = is_fp32_dest_acc_en ? InstrModLoadStore::FP32 : InstrModLoadStore::FP16B;

    // LREG3 = 127.0f (0x42fe)
    TTI_SFPLOADI(p_sfpu::LREG3, SFPLOADI_MOD0_FLOATB, 0x42fe);
    
    // Load 4.791750143340323e-15f ( 0x27aca418 )
    TTI_SFPLOADI(p_sfpu::LREG4, SFPLOADI_MOD0_UPPER, 0x27ac);
    TTI_SFPLOADI(p_sfpu::LREG4, SFPLOADI_MOD0_LOWER, 0xa418);

    // Load 7.839635491371155e-08f ( 0x33a85ada )
    TTI_SFPLOADI(p_sfpu::LREG5, SFPLOADI_MOD0_UPPER, 0x33a8);
    TTI_SFPLOADI(p_sfpu::LREG5, SFPLOADI_MOD0_LOWER, 0x5ada);
    
    // LREG7 = 1/log(2)
    TTI_SFPLOADI(p_sfpu::LREG7, SFPLOADI_MOD0_UPPER, 0x3fb8);
    TTI_SFPLOADI(p_sfpu::LREG7, SFPLOADI_MOD0_LOWER, 0xaa3b);

    // Load 1.0017248f (0x3f803885)
    //TTI_SFPLOADI(p_sfpu::LREG6, SFPLOADI_MOD0_UPPER, 0x3f80);
    //TTI_SFPLOADI(p_sfpu::LREG6, SFPLOADI_MOD0_LOWER, 0x3885);
    TTI_SFPLOADI(p_sfpu::LREG6, SFPLOADI_MOD0_FLOATA, 0x3c02);
    
    // LReg[2] = 255.f
    TTI_SFPLOADI(p_sfpu::LREG2, SFPLOADI_MOD0_FLOATB, 0x437f);

    for (uint32_t i = 0; i < ITERATIONS; i++) {
        
        TTI_SFPLOAD(p_sfpu::LREG0, input_type, ADDR_MOD_7, 0);

        // LREG0 = LREG0 * 1/log(2) + 127.0f
        // LREG0 = LREG0 * LREG7 + LREG2
        TTI_SFPMAD(p_sfpu::LREG0, p_sfpu::LREG7, p_sfpu::LREG3, p_sfpu::LREG0, 0);

        // Since LReg[9] (= 0) is a fixed register, it can not be used for SFPSWAP
        // Instead, we copy LREG9 (LCONST_0) to LREG3 manually
        TTI_SFPMOV(0, p_sfpu::LCONST_0, p_sfpu::LREG1, 0);
	
        // Clamp using min/max
        TTI_SFPSWAP(0, p_sfpu::LREG0, p_sfpu::LREG1, SFPSWAP_MOD1_VEC_MIN_MAX);
        TTI_SFPSWAP(0, p_sfpu::LREG2, p_sfpu::LREG0, SFPSWAP_MOD1_VEC_MIN_MAX);

        // _float_to_int32_for_exp21f_
        TTI_SFPEXEXP(0, p_sfpu::LREG0, p_sfpu::LREG1, 0); // exp = exexp(val)
        TTI_SFPEXMAN(0, p_sfpu::LREG0, p_sfpu::LREG0, 0); // man = exman8(val)
        TTI_SFPSHFT(0, p_sfpu::LREG1, p_sfpu::LREG0, 0); // man = man << exp

        
        // extract fractional part
        TTI_SFPEXMAN(0, p_sfpu::LREG0, p_sfpu::LREG1, SFPEXMAN_MOD1_PAD9);

        // // Convert frac to float32
        // frac = sfpi::int32_to_float(frac, 0)
        constexpr unsigned SFPCAST_MOD1_SM32_TO_FP32_RNE = 0;
        TTI_SFPCAST(p_sfpu::LREG1, p_sfpu::LREG1, SFPCAST_MOD1_SM32_TO_FP32_RNE);

        // Refine approximation of 2**(x_f)
        // ACC = 7.839635491371155e-08f + 4.791750143340323e-15f * frac 
        TTI_SFPMAD(p_sfpu::LREG1, p_sfpu::LREG4, p_sfpu::LREG5, p_sfpu::LREG2, 0);
        TTI_SFPNOP;

        // frac = 1.0017248f + frac * ACC
        TTI_SFPMAD(p_sfpu::LREG2, p_sfpu::LREG1, p_sfpu::LREG6, p_sfpu::LREG1, 0);

        // LReg[2] = 255.f (for next iteration)
        // (Instruction latency hidden by SFPMAD)
        TTI_SFPLOADI(p_sfpu::LREG2, SFPLOADI_MOD0_FLOATB, 0x437f);
        
        constexpr unsigned SFPSETEXP_MOD1_ARG_IMM = 1;
        constexpr unsigned SFPSETEXP_MOD1_ARG_EXPONENT = 2;
        TTI_SFPSETEXP(0, p_sfpu::LREG1, p_sfpu::LREG0, SFPSETEXP_MOD1_ARG_EXPONENT);
	
        if constexpr (!is_fp32_dest_acc_en) {
            constexpr unsigned SFPSTOCHRND_RND_NEAREST = 0;
            TTI_SFP_STOCH_RND(SFPSTOCHRND_RND_NEAREST, 0, p_sfpu::LREG0, p_sfpu::LREG0, p_sfpu::LREG0, SFPSTOCHRND_MOD1_FP32_TO_FP16B);
        }

        TTI_SFPSTORE(p_sfpu::LREG0, input_type, ADDR_MOD_6, 0);
	    //sfpi::dst_reg++;
    }
}
#elif defined(ARCH_BLACKHOLE)
template <bool is_fp32_dest_acc_en, int ITERATIONS>
void calculate_tti_kernel() {

    constexpr uint32_t input_type = is_fp32_dest_acc_en ? InstrModLoadStore::FP32 : InstrModLoadStore::FP16B;

    // LREG3 = 127.0f (0x42fe)
    TTI_SFPLOADI(p_sfpu::LREG3, SFPLOADI_MOD0_FLOATB, 0x42fe);
    
    // Load 4.791750143340323e-15f ( 0x27aca418 )
    TTI_SFPLOADI(p_sfpu::LREG4, SFPLOADI_MOD0_UPPER, 0x27ac);
    TTI_SFPLOADI(p_sfpu::LREG4, SFPLOADI_MOD0_LOWER, 0xa418);

    // Load 7.839635491371155e-08f ( 0x33a85ada )
    TTI_SFPLOADI(p_sfpu::LREG5, SFPLOADI_MOD0_UPPER, 0x33a8);
    TTI_SFPLOADI(p_sfpu::LREG5, SFPLOADI_MOD0_LOWER, 0x5ada);
    
    // LREG7 = 1/log(2)
    TTI_SFPLOADI(p_sfpu::LREG7, SFPLOADI_MOD0_UPPER, 0x3fb8);
    TTI_SFPLOADI(p_sfpu::LREG7, SFPLOADI_MOD0_LOWER, 0xaa3b);

    // Load 1.0017248f (0x3f803885)
    //TTI_SFPLOADI(p_sfpu::LREG6, SFPLOADI_MOD0_UPPER, 0x3f80);
    //TTI_SFPLOADI(p_sfpu::LREG6, SFPLOADI_MOD0_LOWER, 0x3885);
    TTI_SFPLOADI(p_sfpu::LREG6, SFPLOADI_MOD0_FLOATA, 0x3c02);
    
    // LReg[2] = 255.f
    TTI_SFPLOADI(p_sfpu::LREG2, SFPLOADI_MOD0_FLOATB, 0x437f);
    
    for (uint32_t i = 0; i < ITERATIONS; i++) {
        
        TTI_SFPLOAD(p_sfpu::LREG0, input_type, ADDR_MOD_7, 0);

        // LREG0 = LREG0 * 1/log(2) + 127.0f
        // LREG0 = LREG0 * LREG7 + LREG2
        TTI_SFPMAD(p_sfpu::LREG0, p_sfpu::LREG7, p_sfpu::LREG3, p_sfpu::LREG0, 0);

        // Since LReg[9] (= 0) is a fixed register, it can not be used for SFPSWAP
        // Instead, we copy LREG9 (LCONST_0) to LREG3 manually
        TTI_SFPMOV(0, p_sfpu::LCONST_0, p_sfpu::LREG1, 0);
        constexpr unsigned SFPLE_MOD1_SET_VD = 8;
        TTI_SFPLE(0, p_sfpu::LREG0, p_sfpu::LREG1, SFPLE_MOD1_SET_VD); // LREG1 = LREG0 > 0 ? -1 : 0
        constexpr unsigned SFPAND_MOD1_USE_VB = 1;
        TTI_SFPAND(p_sfpu::LREG0, p_sfpu::LREG1, p_sfpu::LREG0, SFPAND_MOD1_USE_VB); // clamp negative values to 0

        // Clamp using min/max
        // TTI_SFPSWAP(0, p_sfpu::LREG0, p_sfpu::LREG1, SFPSWAP_MOD1_VEC_MIN_MAX);
        TTI_SFPSWAP(0, p_sfpu::LREG2, p_sfpu::LREG0, SFPSWAP_MOD1_VEC_MIN_MAX);

        // _float_to_int32_for_exp21f_
        TTI_SFPEXEXP(0, p_sfpu::LREG0, p_sfpu::LREG1, 0); // exp = exexp(val)
        TTI_SFPEXMAN(0, p_sfpu::LREG0, p_sfpu::LREG0, 0); // man = exman8(val)
        TTI_SFPSHFT(0, p_sfpu::LREG1, p_sfpu::LREG0, 0); // man = man << exp

        // extract fractional part
        TTI_SFPEXMAN(0, p_sfpu::LREG0, p_sfpu::LREG1, SFPEXMAN_MOD1_PAD9);

        // // Convert frac to float32
        // frac = sfpi::int32_to_float(frac, 0)
        constexpr unsigned SFPCAST_MOD1_SM32_TO_FP32_RNE = 0;
        TTI_SFPCAST(p_sfpu::LREG1, p_sfpu::LREG1, SFPCAST_MOD1_SM32_TO_FP32_RNE);

        // Refine approximation of 2**(x_f)
        // ACC = 7.839635491371155e-08f + 4.791750143340323e-15f * frac 
        TTI_SFPMAD(p_sfpu::LREG1, p_sfpu::LREG4, p_sfpu::LREG5, p_sfpu::LREG2, 0);
        TTI_SFPNOP;

        // frac = 1.0017248f + frac * ACC
        TTI_SFPMAD(p_sfpu::LREG2, p_sfpu::LREG1, p_sfpu::LREG6, p_sfpu::LREG1, 0);

        // LReg[2] = 255.f (for next iteration)
        // (Instruction latency hidden by SFPMAD)
        TTI_SFPLOADI(p_sfpu::LREG2, SFPLOADI_MOD0_FLOATB, 0x437f);
        
        constexpr unsigned SFPSETEXP_MOD1_ARG_IMM = 1;
        constexpr unsigned SFPSETEXP_MOD1_ARG_EXPONENT = 2;
        TTI_SFPSETEXP(0, p_sfpu::LREG1, p_sfpu::LREG0, SFPSETEXP_MOD1_ARG_EXPONENT);
	
        if constexpr (!is_fp32_dest_acc_en) {
            constexpr unsigned SFPSTOCHRND_RND_NEAREST = 0;
            TTI_SFP_STOCH_RND(SFPSTOCHRND_RND_NEAREST, 0, p_sfpu::LREG0, p_sfpu::LREG0, p_sfpu::LREG0, SFPSTOCHRND_MOD1_FP32_TO_FP16B);
        }

        TTI_SFPSTORE(p_sfpu::LREG0, input_type, ADDR_MOD_6, 0);
	    //sfpi::dst_reg++;
    }
}
#endif // ARCH_WORMHOLE

template <bool is_fp32_dest_acc_en>
void calculate_tti_kernel_init() {

    // Set ADDR MODE 6 to increment on SFPSTORE
    addr_mod_t {
	.srca = {.incr = 0},
	.srcb = {.incr = 0},
	.dest = {.incr = 2},
    }
	.set(ADDR_MOD_6);

    
}
