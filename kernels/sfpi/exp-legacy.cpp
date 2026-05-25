template <bool is_fp32_dest_acc_en = false>
sfpi_inline sfpi::vFloat calculate_sfpi_kernel(sfpi::vFloat val) {
    return _sfpu_exp_21f_bf16_<is_fp32_dest_acc_en>(val);
}



template <bool is_fp32_dest_acc_en>
void calculate_sfpi_kernel_init() {
}