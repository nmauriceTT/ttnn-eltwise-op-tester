template <bool is_fp32_acc_to_dest_mode = true>
sfpi_inline sfpi::vFloat calculate_sfpi_kernel(sfpi::vFloat x) {

    sfpi::vFloat a = x + 2.8f;

    // Branchless clamp using vec_min_max:
    // vec_min_max(a, b) puts min in a, max in b
    sfpi::vFloat low = 0.f;
    sfpi::vFloat high = 5.0f;
    sfpi::vec_min_max(low, a);   // a = max(a, 0.0)
    sfpi::vec_min_max(a, high);  // a = min(a, 5.0)
     
    sfpi::vFloat result = x * a * 0.2f;

    if constexpr (!is_fp32_acc_to_dest_mode) {
        result = sfpi::convert<sfpi::vFloat16b>(result, sfpi::RoundMode::NearestEven);
    }

    return result;
}

inline void calculate_sfpi_kernel_init()
{
}