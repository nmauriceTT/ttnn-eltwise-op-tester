"""
Cumsum accuracy vs tensor length.

For a tensor of shape (4096, 32, 32), run ttnn.cumsum once along dim=0.
For each N in [1, 4096], read slice [N-1] and compare against a float32
PyTorch golden to measure how error accumulates with increasing length.

Three input distributions × two dtypes (bfloat16, float32) are evaluated.

Usage:
    python eval_cumsum_accuracy.py [--output accuracy_results/cumsum_accuracy.csv]

Outputs:
    accuracy_results/cumsum_accuracy.csv
    accuracy_results/cumsum_accuracy_summary_N1000.csv
    accuracy_results/cumsum_accuracy_bfloat16.png
    accuracy_results/cumsum_accuracy_float32.png
"""

from __future__ import annotations

import argparse
import math
import os

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import torch
import ttnn

from models.common.utility_functions import ulp


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SHAPE = (4096, 32, 32)   # run cumsum once; read N-th slice for "length N"
DIM   = 0
SEED  = 42

TORCH_TO_TTNN = {
    # torch.bfloat16: ttnn.bfloat16,
    torch.float32:  ttnn.float32,
}

DISTRIBUTIONS: dict[str, callable] = {
    "ones":      lambda: torch.ones(*SHAPE),
    "randn":     lambda: torch.randn(*SHAPE),
    "randn+0.5": lambda: torch.randn(*SHAPE) + 0.5,
}

EPSILON = 1e-6   # guard against division-by-zero in relative error


# ---------------------------------------------------------------------------
# Metrics for a single slice pair
# ---------------------------------------------------------------------------

def compute_slice_metrics(
    slice_result: torch.Tensor,   # float32, shape (32, 32)
    slice_golden: torch.Tensor,   # float32, shape (32, 32)
    out_dtype: torch.dtype,
) -> dict[str, float]:
    """Return rel_error, ulp_error, psnr for one slice."""
    abs_err = (slice_result - slice_golden).abs()

    # Mean relative error (guard peak against zero)
    denom_rel = slice_golden.abs().clamp(min=EPSILON)
    rel_error  = (abs_err / denom_rel).mean().item()

    # Mean ULP error — ULP is computed in the output dtype
    golden_in_out_dtype = slice_golden.to(out_dtype)
    ulp_vals = ulp(golden_in_out_dtype).float().clamp(min=torch.finfo(torch.float32).tiny)
    ulp_error = (abs_err / ulp_vals).max().item()

    # PSNR
    mse  = (abs_err ** 2).mean().item()
    peak = slice_golden.abs().max().item()
    if mse == 0.0:
        psnr = float("inf")
    elif peak == 0.0:
        psnr = float("-inf")
    else:
        psnr = 20.0 * math.log10(peak) - 10.0 * math.log10(mse)

    return {"rel_error": rel_error, "ulp_error": ulp_error, "psnr": psnr}


# ---------------------------------------------------------------------------
# Data collection
# ---------------------------------------------------------------------------

def collect(device: ttnn.Device) -> pd.DataFrame:
    rows: list[dict] = []
    torch.manual_seed(SEED)

    for torch_dtype, ttnn_dtype in TORCH_TO_TTNN.items():
        dtype_name = str(torch_dtype).replace("torch.", "")

        for dist_name, make_input in DISTRIBUTIONS.items():
            print(f"\n  [{dtype_name} / {dist_name}]", flush=True)

            # --- inputs ---
            raw    = make_input().to(torch_dtype)
            golden = torch.cumsum(raw.float(), dim=DIM)   # float32 reference

            # --- ttnn cumsum ---
            tt_in  = ttnn.from_torch(raw, dtype=ttnn_dtype, layout=ttnn.TILE_LAYOUT, device=device)
            tt_out = ttnn.cumsum(tt_in, dim=DIM, dtype=ttnn_dtype)
            result = ttnn.to_torch(tt_out).float()        # float32 for metrics

            # --- per-length metrics (read N-th slice) ---
            for N in range(1, SHAPE[DIM] + 1):
                s_result = result[N - 1]    # (32, 32)
                s_golden = golden[N - 1]    # (32, 32)
                metrics  = compute_slice_metrics(s_result, s_golden, torch_dtype)
                rows.append({"dtype": dtype_name, "distribution": dist_name, "N": N, **metrics})

            print(f"    done — last-slice rel_err={rows[-1]['rel_error']:.3e}  "
                  f"ulp={rows[-1]['ulp_error']:.2f}  psnr={rows[-1]['psnr']:.1f} dB")

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Summary CSV
# ---------------------------------------------------------------------------

def save_summary(df: pd.DataFrame, output_dir: str) -> None:
    """Save a per-(dtype, distribution) summary for the N=1…1000 window.

    Columns per metric:
      avg_<metric>   — mean over N=1…1000
      n1000_<metric> — value at the 1000th slice
      last_<metric>  — value at the final slice (N=SHAPE[DIM])
    """
    metrics = ["rel_error", "ulp_error", "psnr"]
    key     = ["dtype", "distribution"]
    sub     = df[df["N"] <= 1000]
    agg     = sub.groupby(key)[metrics].mean().add_prefix("avg_")
    at_1000 = df[df["N"] == 1000].set_index(key)[metrics].add_prefix("n1000_")
    at_last = df[df["N"] == df["N"].max()].set_index(key)[metrics].add_prefix("last_")
    summary = agg.join(at_1000).join(at_last)
    path    = os.path.join(output_dir, "cumsum_accuracy_summary_N1000.csv")
    summary.to_csv(path)
    print(f"  Summary saved: {path}")


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot(df: pd.DataFrame, output_dir: str) -> None:
    metrics = [
        # (column name, ylabel, log scale)
        ("rel_error", "Mean relative error",  True),
        ("ulp_error", "Max ULP error",        True),
        ("psnr",      "PSNR (dB) \n(higher is better)",             False),
    ]

    for torch_dtype in TORCH_TO_TTNN:
        dtype_name = str(torch_dtype).replace("torch.", "")
        sub = df[df["dtype"] == dtype_name]

        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        fig.suptitle(f"ttnn.cumsum accuracy vs length  [{dtype_name}]", fontsize=13)

        for ax, (col, ylabel, log_scale) in zip(axes, metrics):
            sns.lineplot(data=sub, x="N", y=col, hue="distribution", ax=ax)
            ax.set_xlabel("Cumsum length (N)")
            ax.set_ylabel(ylabel)
            ax.set_title(ylabel)
            if log_scale:
                ax.set_yscale("log")
                # Ensure y=1 is visible and draw a reference line
                ymin, ymax = ax.get_ylim()
                ax.set_ylim(min(ymin, 0.5), max(ymax, 2.0))
                ax.axhline(y=1, color="black", linestyle="--", linewidth=1.0, alpha=0.6, label="y=1")
                # Re-draw legend to include the y=1 line
                handles, labels = ax.get_legend_handles_labels()
                ax.legend(handles=handles, labels=labels, title="distribution")
            else:
                ax.set_ylim(0,)
                ax.legend(title="distribution")

        fig.tight_layout()
        out_path = os.path.join(output_dir, f"cumsum_accuracy_{dtype_name}.png")
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        print(f"  Plot saved: {out_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        default="accuracy_results/cumsum_accuracy.csv",
        help="Output CSV path (default: accuracy_results/cumsum_accuracy.csv)",
    )
    args = parser.parse_args()

    output_dir = os.path.dirname(args.output) or "."
    os.makedirs(output_dir, exist_ok=True)

    device = ttnn.open_device(device_id=0)
    try:
        print("Collecting cumsum accuracy data …")
        df = collect(device)
    finally:
        ttnn.close_device(device)

    df.to_csv(args.output, index=False)
    print(f"\nCSV saved: {args.output}")

    print("Generating summary …")
    save_summary(df, output_dir)

    print("Generating plots …")
    plot(df, output_dir)
    print("Done.")


if __name__ == "__main__":
    main()
