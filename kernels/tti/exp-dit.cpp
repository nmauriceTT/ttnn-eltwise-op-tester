
#if defined(TRISC_MATH) || defined(TRISC_PACK)
namespace custom {
constexpr auto bits = [](float x) constexpr { return __builtin_bit_cast(std::uint32_t, x); };
constexpr auto lo16 = [](float x) constexpr { return static_cast<std::uint16_t>(bits(x) & 0xFFFFu); };
constexpr auto hi16 = [](float x) constexpr { return static_cast<std::uint16_t>(bits(x) >> 16); };

#ifdef ARCH_WORMHOLE
#define ADDR_MOD_X ADDR_MOD_3
#else
#define ADDR_MOD_X ADDR_MOD_7
#endif

ALWI void INSERT_SFPNOP() {
#ifdef ARCH_WORMHOLE
    TTI_SFPNOP;
#endif
}

template <bool USE_SFPARECIP_INSTR, int POLY_DEGREE>
constexpr bool can_preload_ln2_constants() {
#ifdef ARCH_WORMHOLE
    return false;
#else
    return (USE_SFPARECIP_INSTR || POLY_DEGREE == 1 || POLY_DEGREE == 2);
#endif
}

/**
 * Computes exp(x) using polynomial approximation after range reduction.
 *
 * Scales by configured factor, then reduces to exp(r) * 2^k
 * where r = x - k*ln(2). Uses either SFPARECIP instruction or multi-term polynomial (degree 1-4)
 * to compute exp(r), then reconstructs full result via exponent manipulation,
 * clamping the exponent to handle large positive or negative inputs.
 *
 * @tparam USE_SFPARECIP_INSTR Use hardware SFPARECIP instruction (true) or polynomial evaluation (false). Only
 * supported on Blackhole.
 * @tparam SCALE_EN Apply scaling factor from LREG to input values
 * @tparam ITERATIONS Number of 32-element vectors to process per tile
 * @tparam POLY_DEGREE Polynomial degree (1-4) when USE_SFPARECIP_INSTR=false; higher improves accuracy
 * @tparam IS_FP32_DEST_ACC_EN Float32 accumulation to dest register enabled.
 * @tparam SCALE_BF16 Bfloat16 scale factor represented as uint16_t.
 */
template <
    bool SCALE_EN,
    int ITERATIONS,
    bool USE_SFPARECIP_INSTR,
    int POLY_DEGREE,
    bool IS_FP32_DEST_ACC_EN,
    uint16_t SCALE_BF16>
void calculate_exponential_polynomial() {
    addr_mod_t{
        .srca = {.incr = 0},
        .srcb = {.incr = 0},
        .dest = {.incr = 0},
    }
        .set(ADDR_MOD_7);

    constexpr float LN2_RECIP = 1.44269504088896340736f;  // 1/ln(2)
    constexpr float M_LN2 = -0.69314718055994530942f;     // -ln(2)

    if constexpr (!USE_SFPARECIP_INSTR) {
        static_assert(POLY_DEGREE >= 1 && POLY_DEGREE <= 4);

        // Evaluate polynomial f(x) = c0 + c1 * x + c2 * x^2 + ... using Horner's method.
        constexpr float c0 = (POLY_DEGREE == 1)   ? 1.03022936050163882354355235184958220293399209290987f
                             : (POLY_DEGREE == 2) ? 0.999848792924395313327307061545061386175496934006f
                             : (POLY_DEGREE == 3) ? 0.99992449655091231753798502608929170703152709521188f
                                                  : 1.0000001510806179002040134468008959160576106495165f;
        constexpr float c1 = (POLY_DEGREE == 1)   ? 1.0201394465967894800285756834161653337107187804001f
                             : (POLY_DEGREE == 2) ? 1.01508760098521056684783640695492761469306929535975f
                             : (POLY_DEGREE == 3) ? 0.99993960415029750534472970577402987498389428593233f
                                                  : 0.99996228117047652035114096488703457970402030983204f;
        constexpr float c2 = (POLY_DEGREE == 2)   ? 0.50628367056745568861842335616023694454759126020461f
                             : (POLY_DEGREE == 3) ? 0.50502329058055065591138054839814880512001604099324f
                                                  : 0.49998365704615426417337683145647067790385638465486f;
        constexpr float c3 = (POLY_DEGREE == 3) ? 0.16817330195731531429790827442800245470170482723302f
                                                : 0.16792157982882225102649214918047336097544632172075f;
        constexpr float c4 = 4.1959439860014343843000081999668024587178974865521e-2;

        // Load polynomial coefficients.
        if constexpr (POLY_DEGREE >= 4) {
            TTI_SFPLOADI(p_sfpu::LREG3, 0xA, lo16(c4));
            TTI_SFPLOADI(p_sfpu::LREG3, 0x8, hi16(c4));
        }
        if constexpr (POLY_DEGREE >= 3) {
            TTI_SFPLOADI(p_sfpu::LREG4, 0xA, lo16(c3));
            TTI_SFPLOADI(p_sfpu::LREG4, 0x8, hi16(c3));
        }
        if constexpr (POLY_DEGREE >= 2) {
            TTI_SFPLOADI(p_sfpu::LREG5, 0xA, lo16(c2));
            TTI_SFPLOADI(p_sfpu::LREG5, 0x8, hi16(c2));
        }
        if constexpr (POLY_DEGREE >= 1) {
            TTI_SFPLOADI(p_sfpu::LREG6, 0xA, lo16(c1));
            TTI_SFPLOADI(p_sfpu::LREG6, 0x8, hi16(c1));
            TTI_SFPLOADI(p_sfpu::LREG7, 0xA, lo16(c0));
            TTI_SFPLOADI(p_sfpu::LREG7, 0x8, hi16(c0));
        }
    }

    if constexpr (can_preload_ln2_constants<USE_SFPARECIP_INSTR, POLY_DEGREE>()) {
        TTI_SFPLOADI(p_sfpu::LREG3, 0xA, lo16(LN2_RECIP));
        TTI_SFPLOADI(p_sfpu::LREG3, 0x8, hi16(LN2_RECIP));
        TTI_SFPLOADI(p_sfpu::LREG4, 0xA, lo16(M_LN2));
        TTI_SFPLOADI(p_sfpu::LREG4, 0x8, hi16(M_LN2));
    }

    for (int d = 0; d < ITERATIONS; d++) {
        // Load the input.
        constexpr uint8_t input_type = IS_FP32_DEST_ACC_EN ? InstrModLoadStore::FP32 : InstrModLoadStore::FP16B;
        TTI_SFPLOAD(p_sfpu::LREG2, input_type, ADDR_MOD_X, 0);

        if constexpr (SCALE_EN) {
            TTI_SFPLOADI(p_sfpu::LREG0, 0, SCALE_BF16);
            TTI_SFPMAD(p_sfpu::LREG2, p_sfpu::LREG0, p_sfpu::LCONST_0, p_sfpu::LREG2, 0);
            INSERT_SFPNOP();
        }

        // Multiply by 1/ln(2) and round.
        if constexpr (can_preload_ln2_constants<USE_SFPARECIP_INSTR, POLY_DEGREE>()) {
            TTI_SFPMAD(p_sfpu::LREG2, p_sfpu::LREG3, p_sfpu::LCONST_0, p_sfpu::LREG0, 0);
        } else {
            TTI_SFPLOADI(p_sfpu::LREG1, 0xA, lo16(LN2_RECIP));
            TTI_SFPLOADI(p_sfpu::LREG1, 0x8, hi16(LN2_RECIP));
            TTI_SFPMAD(p_sfpu::LREG2, p_sfpu::LREG1, p_sfpu::LCONST_0, p_sfpu::LREG0, 0);
        }
        INSERT_SFPNOP();
        TTI_SFP_STOCH_RND(
            0, 0, 0, p_sfpu::LREG0, p_sfpu::LREG1, sfpi::SFPSTOCHRND_MOD1_FP32_TO_INT8);  // Clamp to [-127,+127].
        TTI_SFPCAST(p_sfpu::LREG1, p_sfpu::LREG1, 0);

        if constexpr (USE_SFPARECIP_INSTR) {
#ifdef ARCH_BLACKHOLE
            // Calculate floor(x) by setting v=v-1 if v>u.
            TTI_SFPGT(0, p_sfpu::LREG0, p_sfpu::LREG1, 1);                                    // SFPGT_MOD1_SET_CC
            TTI_SFPMAD(p_sfpu::LCONST_1, p_sfpu::LREG1, p_sfpu::LCONST_1, p_sfpu::LREG1, 2);  // SFPMAD_MOD1_NEGATE_VC
            TTI_SFPENCC(0, 0, 0, 0);

            // Calculate exp(x - k*ln2).
            TTI_SFPMAD(p_sfpu::LREG1, p_sfpu::LREG4, p_sfpu::LREG2, p_sfpu::LREG0, 0);
            TTI_SFPARECIP(0, p_sfpu::LREG0, p_sfpu::LREG0, 2);
#else
            ASSERT(false);  // TTI_SFPARECIP instruction only supported on Blackhole".
#endif
        } else {
            if constexpr (can_preload_ln2_constants<USE_SFPARECIP_INSTR, POLY_DEGREE>()) {
                TTI_SFPMAD(p_sfpu::LREG1, p_sfpu::LREG4, p_sfpu::LREG2, p_sfpu::LREG0, 0);
            } else {
                TTI_SFPLOADI(p_sfpu::LREG0, 0xA, lo16(M_LN2));
                TTI_SFPLOADI(p_sfpu::LREG0, 0x8, hi16(M_LN2));
                TTI_SFPMAD(p_sfpu::LREG1, p_sfpu::LREG0, p_sfpu::LREG2, p_sfpu::LREG0, 0);
            }
            INSERT_SFPNOP();

            // Calculate polynomial.
            if constexpr (POLY_DEGREE == 1) {
                TTI_SFPMAD(p_sfpu::LREG0, p_sfpu::LREG6, p_sfpu::LREG7, p_sfpu::LREG0, 0);
            } else if constexpr (POLY_DEGREE == 2) {
                TTI_SFPMAD(p_sfpu::LREG0, p_sfpu::LREG5, p_sfpu::LREG6, p_sfpu::LREG2, 0);
                INSERT_SFPNOP();
                TTI_SFPMAD(p_sfpu::LREG0, p_sfpu::LREG2, p_sfpu::LREG7, p_sfpu::LREG0, 0);
            } else if constexpr (POLY_DEGREE == 3) {
                TTI_SFPMAD(p_sfpu::LREG0, p_sfpu::LREG4, p_sfpu::LREG5, p_sfpu::LREG2, 0);
                INSERT_SFPNOP();
                TTI_SFPMAD(p_sfpu::LREG0, p_sfpu::LREG2, p_sfpu::LREG6, p_sfpu::LREG2, 0);
                INSERT_SFPNOP();
                TTI_SFPMAD(p_sfpu::LREG0, p_sfpu::LREG2, p_sfpu::LREG7, p_sfpu::LREG0, 0);
            } else {  // degree 4.
                TTI_SFPMAD(p_sfpu::LREG0, p_sfpu::LREG3, p_sfpu::LREG4, p_sfpu::LREG2, 0);
                INSERT_SFPNOP();
                TTI_SFPMAD(p_sfpu::LREG0, p_sfpu::LREG2, p_sfpu::LREG5, p_sfpu::LREG2, 0);
                INSERT_SFPNOP();
                TTI_SFPMAD(p_sfpu::LREG0, p_sfpu::LREG2, p_sfpu::LREG6, p_sfpu::LREG2, 0);
                INSERT_SFPNOP();
                TTI_SFPMAD(p_sfpu::LREG0, p_sfpu::LREG2, p_sfpu::LREG7, p_sfpu::LREG0, 0);
            }
            INSERT_SFPNOP();
        }

        // Multiply by 2^k.
        TT_SFPADDI(0x42fe, p_sfpu::LREG1, 0);  // Add 127.
        INSERT_SFPNOP();
        TTI_SFP_STOCH_RND(0, 0, 0, p_sfpu::LREG1, p_sfpu::LREG2, sfpi::SFPSTOCHRND_MOD1_FP32_TO_UINT8);
        TTI_SFPSETEXP(0, p_sfpu::LCONST_0, p_sfpu::LREG2, 0);
        TTI_SFPMAD(p_sfpu::LREG0, p_sfpu::LREG2, p_sfpu::LCONST_0, p_sfpu::LREG2, 0);
        INSERT_SFPNOP();

        // Handle underflow: if k == 0, exp(x) = 0 (fixes -inf case).
        TTI_SFPSETCC(0, p_sfpu::LREG1, 0, 6);  // Set LaneFlags = (LREG1 == 0) and enable CC.
        TTI_SFPLOADI(p_sfpu::LREG2, 0, 0);     // LREG2 = 0 ONLY for lanes where LREG1 == 0.
        TTI_SFPENCC(0, 0, 0, 0);               // Disable CC and clear LaneFlags - ALL lanes active again.

        // Store the result.
        if constexpr (!IS_FP32_DEST_ACC_EN) {
            // LRegs work on float32 data. If DST is bfloat16 then SFPSTORE will truncate it
            // so convert to bfloat16 using round-to-nearest-even.
            TTI_SFP_STOCH_RND(0, 0, 0, p_sfpu::LREG2, p_sfpu::LREG2, sfpi::SFPSTOCHRND_MOD1_FP32_TO_FP16B);
        }
        TTI_SFPSTORE(p_sfpu::LREG2, input_type, ADDR_MOD_X, 0);
        TTI_INCRWC(0, 4, 0, 0);  // Skip odd columns.
    }
}
}

template <bool is_fp32_dest_acc_en, int ITERATIONS>
void calculate_tti_kernel() {

    constexpr uint16_t scale_bf16 = 0x8000;
    constexpr int polynomial_degree = DST_ACCUM_MODE ? 2 : 2;
        custom::calculate_exponential_polynomial<
            false,
            ITERATIONS,
            false,
            polynomial_degree,
            false,
            scale_bf16>();

}

template <bool is_fp32_dest_acc_en>
void calculate_tti_kernel_init() {
    
}

#endif  // defined(TRISC_MATH) || defined(TRISC_PACK)