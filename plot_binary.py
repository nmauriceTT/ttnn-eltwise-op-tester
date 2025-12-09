import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns
import pandas as pd
import numpy as np
import sys
import os
import glob
import json

from matplotlib.colors import LogNorm, BoundaryNorm

from plot_params_parser import (
    parse_plot_config,
    generate_plot_config_hashes,
    save_plot_config_hashes,
    load_plot_config_hashes,
    plot_all,
)


plt.rcParams["svg.fonttype"] = "none"  # Make text editing in SVG easier


def load_csv(filename):
    """Load CSV file with binary operation accuracy results"""
    return pd.read_csv(filename, sep=",", index_col=False, skipinitialspace=True)



def plot_heatmap(plot_entry):
    """
    Create a heatmap from binary operation accuracy data using parameters from JSON config.
    Similar to the plot function in plot_accuracy.py but adapted for heatmaps.

    Args:
        plot_entry: Dictionary containing plot configuration from binary-plots.json
    """


    try:

        # Extract data from plot_entry
        data = plot_entry["data"]
        plot_params = plot_entry["plot_params"]
        operation_name = plot_entry["name"]
        output_paths = plot_entry["outputs"]


        # Get parameters from JSON config
        colormap_config = plot_params.get(
            "colormap",
            [
                {"threshold": 0, "color": "#2E8B57"},
                {"threshold": 2, "color": "#FFD700"},
                {"threshold": 5, "color": "#FF6347"},
                {"threshold": 10, "color": "#8B0000"},
            ],
        )

        alim = plot_params.get("alim", [-1e9, 1e9])
        blim = plot_params.get("blim", [-1e9, 1e9])

        # Extract column names from config
        aname = plot_entry.get("aname", "a")
        bname = plot_entry.get("bname", "b")
        valuename = plot_entry.get("valuename", "max_ulp_error")

        # Create pivot table with a on x-axis, b on y-axis, max_ulp_error as values
        pivot_data = data.pivot(columns=aname, index=bname, values=valuename)

        # Create figure with appropriate size
        plt.figure(figsize=(12, 10))

        # Create custom colormap from JSON config
        # colors = [item["color"] for item in colormap_config]
        # boundaries = [item["threshold"] for item in colormap_config] + [1e9]

        # print(f"BOUNDARIES = {boundaries}")
        # print(f"COLORS = {colors}")

        # cmap = mcolors.ListedColormap(colors)
        # norm = mcolors.BoundaryNorm(boundaries, cmap.N)
        # levels = [1e-6, 1, 2, 3, 4, 5, 10, 100, 100]
        # Collect all threshold values and ensure they're sorted and monotonically increasing
        thresholds = [item["threshold"] for item in colormap_config]
        # Combine 1e-6 with thresholds, remove duplicates, and sort to ensure monotonicity
        levels = sorted(set([1e-6] + thresholds))
        cmap = plt.colormaps[plot_params.get("colormap_name", "PiYG")]
        cmap.set_bad(color="purple")

        norm = BoundaryNorm(levels, ncolors=cmap.N, clip=True)

        # Handle NaN values
        nan_mask = np.isnan(pivot_data.values)

        # Create the color mesh
        x_coords = pivot_data.columns.values
        y_coords = pivot_data.index.values

        # Create meshgrid for pcolormesh
        X, Y = np.meshgrid(x_coords, y_coords)

        # Create the color mesh
        mesh = plt.pcolormesh(X, Y, pivot_data.values, norm=norm, cmap="PuBu_r", shading="auto")

        # Create colorbar
        cbar = plt.colorbar(mesh, norm=norm, label="Max ULP Error", ticks=levels)

        # Set background color for NaN values to Yellow
        plt.gca().set_facecolor("yellow")


        ascale = plot_params.get("ascale", "symlog")
        bscale = plot_params.get("bscale", "symlog")
        
        # If using log scale with negative limits, switch to symlog
        if ascale == "log" and alim[0] < 0:
            print(f"Warning: log scale cannot handle negative values. Switching to symlog for x-axis.")
            ascale = "symlog"
        if bscale == "log" and blim[0] < 0:
            print(f"Warning: log scale cannot handle negative values. Switching to symlog for y-axis.")
            bscale = "symlog"
        
        plt.gca().set_xscale(ascale)
        plt.gca().set_yscale(bscale)

        # Use limits from JSON config
        plt.gca().set_xlim(alim[0], alim[1])
        plt.gca().set_ylim(blim[0], blim[1])

        print(f"Xlim: {plt.gca().get_xlim()}")
        print(f"Ylim: {plt.gca().get_ylim()}")
        print(f"alim = {alim}, blim = {blim}")

        # Add reference lines
        if "horizontal_lines" in plot_params:
            for horizontal_line in plot_params["horizontal_lines"]:
                plt.axhline(y=horizontal_line[0], color="red", linestyle="--", linewidth=1.5, label=horizontal_line[1], alpha=0.5)
                plt.text(x=plt.gca().get_xlim()[0], y=horizontal_line[0], s=horizontal_line[1], color="red", va="bottom", ha="left")

        if "vertical_lines" in plot_params:
            for vertical_line in plot_params["vertical_lines"]:
                plt.axvline(x=vertical_line[0], color="red", linestyle="--", linewidth=1.5, label=vertical_line[1], alpha=0.5)
                plt.text(x=vertical_line[0], y=plt.gca().get_ylim()[1] * 0.9, s=vertical_line[1], color="red", va="top", ha="left")

        # Set title and labels

        title = plot_params.get("title", None)
        short_name = plot_entry["name"] if "name" in plot_entry else operation_name
        if title is not None:
            title = title.format(short_name)

        plt.title(title + f"\n(Yellow = NaN or Inf)", pad=20)
        plt.xlabel("X")
        plt.ylabel("Y")

        # Adjust layout to prevent label cutoff
        plt.tight_layout()

        # Save the plot to all specified output paths
        for output_path in output_paths:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            plt.savefig(output_path, dpi=300, bbox_inches="tight")

        plt.close()
    except Exception as e:
        import traceback
        print(f"Error plotting {plot_entry['id']}: {e}")
        traceback.print_exc()
        plt.close()


def plot_histogram(plot_entry):
    """
    Create a histogram from binary operation accuracy data showing ULP error distribution.

    Args:
        plot_entry: Dictionary containing plot configuration from binary-plots.json
    """

    try:
        # Extract data from plot_entry
        data = plot_entry["data"]
        plot_params = plot_entry["plot_params"]
        operation_name = plot_entry["name"]
        output_paths = plot_entry["outputs"]

        # Get the value column name (default: max_ulp_error)
        valuename = plot_entry.get("valuename", "max_ulp_error")

        # Extract values, filtering out NaN and infinite values
        values = data[valuename].values
        values = values[np.isfinite(values)]

        if len(values) == 0:
            print(f"Warning: No valid data for histogram {plot_entry['id']}")
            return

        # Create figure
        fig, ax = plt.subplots(figsize=(12, 8))

        # Get histogram parameters from config
        bins = plot_params.get("bins", 50)
        log_scale = plot_params.get("log_scale", True)
        color = plot_params.get("color", "#4C72B0")
        edgecolor = plot_params.get("edgecolor", "white")

        # Calculate statistics
        max_error = np.max(values)
        min_error = np.min(values)
        mean_error = np.mean(values)
        median_error = np.median(values)
        p99_error = np.percentile(values, 99)
        p95_error = np.percentile(values, 95)

        # Create histogram
        if log_scale and min_error > 0:
            # Use log-spaced bins for better visualization of wide ranges
            log_min = np.log10(max(min_error, 1e-10))
            log_max = np.log10(max(max_error, 1))
            bin_edges = np.logspace(log_min, log_max, bins + 1)
            ax.set_xscale("log")
        else:
            # Handle zero/negative values with linear bins
            bin_edges = bins

        # Normalize to percentages using weights
        weights = np.ones_like(values) * (100.0 / len(values))

        counts, bin_edges, patches = ax.hist(
            values,
            bins=bin_edges,
            weights=weights,
            color=color,
            edgecolor=edgecolor,
            alpha=0.8,
            linewidth=0.5,
        )

        # Set y-axis to log scale if specified
        if plot_params.get("log_y", True) and np.max(counts) > 0:
            ax.set_yscale("log")

        # Add vertical lines for statistics
        ax.axvline(x=mean_error, color="#E24A33", linestyle="--", linewidth=2, label=f"Mean: {mean_error:.2f}")
        ax.axvline(x=median_error, color="#348ABD", linestyle="-.", linewidth=2, label=f"Median: {median_error:.2f}")
        ax.axvline(x=p99_error, color="#988ED5", linestyle=":", linewidth=2, label=f"P99: {p99_error:.2f}")
        ax.axvline(x=max_error, color="#8EBA42", linestyle="-", linewidth=2, label=f"Max: {max_error:.2f}")

        # Set title and labels
        title = plot_params.get("title", "ULP Error Distribution of ttnn.{}")
        short_name = plot_entry.get("name", operation_name)
        title = title.format(short_name)

        ax.set_title(title, fontsize=16, pad=15)
        ax.set_xlabel("ULP Error", fontsize=14)
        ax.set_ylabel("Frequency (%)", fontsize=14)
        ax.set_yscale("linear")

        # Add legend
        ax.legend(loc="upper right", fontsize=10, framealpha=0.9)

        # Add statistics text box
        stats_text = (
            f"Total samples: {len(values):,}\n"
            f"Max: {max_error:.2f}\n"
            f"P99: {p99_error:.2f}\n"
            f"P95: {p95_error:.2f}\n"
            f"Mean: {mean_error:.2f}\n"
            f"Median: {median_error:.2f}\n"
            f"Min: {min_error:.2f}"
        )
        props = dict(boxstyle="round", facecolor="wheat", alpha=0.8)
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=10,
                verticalalignment="top", bbox=props)

        # Add grid
        ax.grid(True, alpha=0.3, which="both")

        plt.tight_layout()

        # Save the plot to all specified output paths
        for output_path in output_paths:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            plt.savefig(output_path, dpi=300, bbox_inches="tight")

        plt.close()

    except Exception as e:
        import traceback
        print(f"Error plotting histogram {plot_entry['id']}: {e}")
        traceback.print_exc()
        plt.close()


def preprocess_data(data):
    """
    Remove rows where 'a' or 'b' are either infinite or NaN.
    Assumes columns are named 'a' and 'b'.
    """
    # Remove rows where 'a' or 'b' are NaN or infinite
    mask = (np.isfinite(data["a"])) & (np.isfinite(data["b"]))
    print(f"Preprocessed data shape: {data[mask].shape}")

    data = data[mask]

    # mask_useful = (
    #    (data['a'] > -1e3) & (data['a'] < 1e3) &
    #    (data['b'] > -1e3) & (data['b'] < 1e3)
    # )

    # data = data[mask_useful]

    return data



def sanitize_data(data):
    # Remove data with NaN or infinity (might speedup plotting)
    data = data[(data["a"].notna()) & (np.isfinite(data["a"]))]
    data = data[(data["b"].notna()) & (np.isfinite(data["b"]))]
    data = data.reset_index()

    return data


def main():
    """Main function to process all binary operation CSV files"""
    # Default input and output directories

    sns.set(
        style="ticks",
        rc={
            "axes.grid": True,
            # "axes.edgecolor": None,
            "axes.titlesize": 20,
            "axes.labelsize": 20,
            "xtick.labelsize": 15,
            "ytick.labelsize": 15,
            "font.size": 20,
            "legend.title_fontsize": 20,
            "legend.fontsize": 20,
            "lines.linewidth": 4,
            "axes.linewidth": 1,
            "font.serif": ["Latin Modern Math"],
            "lines.markersize": 8,
            "lines.markeredgecolor": "none",
        },
    )

    input_dir = "accuracy_results/results/binary"
    output_dir = "accuracy_results/plots/binary_heatmaps"

    # Allow command line arguments to override directories
    if len(sys.argv) > 1:
        input_dir = sys.argv[1]
    if len(sys.argv) > 2:
        output_dir = sys.argv[2]

    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        print(f"Create output directory: {output_dir}")
        os.makedirs(output_dir)

    # Create input directory if it doesn't exist
    if not os.path.exists(input_dir):
        print(f"Input directory does not exist: {input_dir}")
        os.makedirs(input_dir)

    # Example 1: Using the original create_heatmap function (hardcoded)
    accuracy_dir = "accuracy_results/results/"
    dest_dir = "accuracy_results/plots"

    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)

    if not os.path.exists(f"{dest_dir}/abs/"):
        os.makedirs(f"{dest_dir}/abs/")

    # plot_all_ops(f"{accuracy_dir}", all_operations, f"{dest_dir}/abs/", highres=False, plot_absolute=True)

    plot_config = parse_plot_config("configs/binary-plots.json")

    # Get time stamp of last modification of this script
    script_mtime = os.path.getmtime(__file__)

    # Separate groups by plot type
    heatmap_groups = []
    histogram_groups = []

    for group in plot_config["groups"]:
        plot_type = group.get("default_params", {}).get("type", "heatmap")
        if plot_type == "histogram":
            histogram_groups.append(group)
        else:
            heatmap_groups.append(group)

    # Process heatmaps
    if heatmap_groups:
        heatmap_config = {"groups": heatmap_groups}
        last_hashes_heatmap = load_plot_config_hashes(f"accuracy_results/binary-plot-hashes.csv")
        
        force_replot_heatmap = False
        if script_mtime > last_hashes_heatmap["last_modified"]:
            force_replot_heatmap = True
        
        current_hashes_heatmap = generate_plot_config_hashes(heatmap_config)
        plot_all(plot_heatmap, heatmap_config, dest_dir, last_hashes_heatmap, current_hashes_heatmap, force_replot_heatmap, sanitize_data)
        save_plot_config_hashes(current_hashes_heatmap, f"accuracy_results/binary-plot-hashes.csv")
        print("All JSON-configured heatmaps generated successfully!")

    # Process histograms
    if histogram_groups:
        histogram_config = {"groups": histogram_groups}
        last_hashes_histogram = load_plot_config_hashes(f"accuracy_results/binary-histogram-hashes.csv")
        
        force_replot_histogram = False
        if script_mtime > last_hashes_histogram["last_modified"]:
            force_replot_histogram = True
        
        current_hashes_histogram = generate_plot_config_hashes(histogram_config)
        plot_all(plot_histogram, histogram_config, dest_dir, last_hashes_histogram, current_hashes_histogram, force_replot_histogram, sanitize_data)
        save_plot_config_hashes(current_hashes_histogram, f"accuracy_results/binary-histogram-hashes.csv")
        print("All JSON-configured histograms generated successfully!")


if __name__ == "__main__":
    main()
