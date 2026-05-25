template <bool is_fp32_dest_acc_en = false>
sfpi_inline sfpi::vFloat calculate_sfpi_kernel(sfpi::vFloat val) {
    sfpi::vFloat result = sfpi::vConst0;
    v_if (val <= val) {
        result = 1.f;
    } v_else {
        result = 0.f;
    }
    v_endif;

    return result;
}

template <bool is_fp32_dest_acc_en>
void calculate_sfpi_kernel_init() {
}