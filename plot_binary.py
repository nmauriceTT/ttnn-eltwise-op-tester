import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.ticker as ticker
import seaborn as sns
import pandas as pd
import numpy as np
import sys
import os
import glob
import json

from matplotlib.colors import LogNorm, BoundaryNorm

from src.plot_params_parser import (
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


def save_histogram_debug_csv(error_values, output_path):
    """
    Save raw histogram data without binning to CSV for debugging.
    
    Args:
        error_values: Array of error values (already filtered for finite values)
        output_path: Path where to save the CSV file
    """
    # Get unique values and their counts
    unique_values, counts = np.unique(error_values, return_counts=True)
    
    # Calculate percentages
    total = len(error_values)
    percentages = (counts / total) * 100
    
    # Calculate cumulative percentages
    cumulative_percentages = np.cumsum(percentages)
    
    # Create DataFrame
    df = pd.DataFrame({
        'ulp_error': unique_values,
        'count': counts,
        'percentage': percentages,
        'cumulative_percentage': cumulative_percentages
    })
    
    # Save to CSV
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False, float_format='%.6f')
    print(f"Saved debug CSV: {output_path}")


def plot_histogram(plot_entry):
    """
    Create a histogram with cumulative distribution from binary operation accuracy data.
    
    Args:
        plot_entry: Dictionary containing plot configuration from binary-plots.json
    """
    try:
        # Extract data from plot_entry
        data = plot_entry["data"]
        plot_params = plot_entry["plot_params"]
        operation_name = plot_entry["name"]
        output_paths = plot_entry["outputs"]
        
        # Extract column name for the error values
        valuename = plot_entry.get("valuename", "max_ulp_error")
        
        # Get error values and remove NaN/Inf
        error_values = data[valuename].values
        error_values = error_values[np.isfinite(error_values)]
        
        if len(error_values) == 0:
            print(f"Warning: No finite error values for {operation_name}")
            return
        
        # Save debug CSV with raw values (no binning)
        # Generate CSV path from the first PNG output path
        # if output_paths:
        #     csv_path = output_paths[0].replace('.png', '_debug.csv')
        #     save_histogram_debug_csv(error_values, csv_path)
        
        # Calculate statistics
        max_error = np.max(error_values)
        mean_error = np.mean(error_values)
        
        # Create figure
        fig, ax = plt.subplots(figsize=(14, 10))
        
        # Determine x-axis range
        x_min = 0
        x_max = max_error * 1.1  # Add 10% padding
        
        # Create x-axis ticks: [1, 2, 5, 10] + powers of 10
        x_ticks = [1, 2, 5, 10]
        power = 1
        while 10**power < x_max:
            power += 1
            x_ticks.append(10**power)
        x_ticks = [x for x in x_ticks if x <= x_max]
        
        # Create histogram bins with meaningful ULP error ranges
        # Use bins like: [0-1, 1-2, 2-5, 5-10, 10-20, 20-50, 50-100, ...]
        bins = [0, 1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000]
        # Extend with powers of 10 if needed
        power = 4
        while 10**power < x_max:
            power += 1
            bins.append(10**power)
        # Only keep bins up to max_error
        bins = [b for b in bins if b <= x_max]
        bins.append(x_max)  # Add the max as final bin edge
        bins = np.array(bins)
        
        # Calculate histogram
        hist_counts, bin_edges = np.histogram(error_values, bins=bins)
        hist_proportion = hist_counts / len(error_values)  # Keep as proportion [0, 1]
        
        # Calculate bin centers and widths (bins have variable widths!)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        bin_widths = bin_edges[1:] - bin_edges[:-1]
        
        # Calculate cumulative distribution
        sorted_errors = np.sort(error_values)
        cumulative_proportion = np.arange(1, len(sorted_errors) + 1) / len(sorted_errors)  # [0, 1]
        
        # Need to handle edge cases for logit scale (0 and 1)
        # Logit scale is only defined for (0, 1), so we need to clip values
        epsilon = 0.0001  # Small value to avoid singularities at 0 and 1
        
        # Debug: print histogram statistics
        non_zero_hist = hist_proportion[hist_proportion > 0]
        if len(non_zero_hist) > 0:
            print(f"Histogram stats: min={np.min(non_zero_hist):.6f}, max={np.max(non_zero_hist):.6f}, mean={np.mean(non_zero_hist):.6f}")
            print(f"Number of non-zero bins: {len(non_zero_hist)}/{len(hist_proportion)}")
            print(f"Sum of all histogram proportions: {np.sum(hist_proportion):.6f} (should be 1.0)")
            print(f"Bin widths: {bin_widths}")
            print(f"Max error: {max_error:.2f}, Mean error: {mean_error:.2f}")
        
        # Set scales FIRST before plotting, so matplotlib interprets coordinates correctly
        ax.set_xscale('asinh')
        ax.set_yscale('logit')
        
        # For logit scale, only plot non-zero bins and clip away from 0 and 1
        mask = hist_proportion > 0
        hist_proportion_nonzero = hist_proportion[mask]
        bin_edges_nonzero_left = bin_edges[:-1][mask]
        bin_edges_nonzero_right = bin_edges[1:][mask]
        
        # Clip histogram proportions for logit scale
        hist_proportion_clipped = np.clip(hist_proportion_nonzero, epsilon, 1 - epsilon)
        
        # Plot histogram as bars - draw each bar individually using rectangles
        from matplotlib.patches import Rectangle
        for i, (left, right, height) in enumerate(zip(bin_edges_nonzero_left, bin_edges_nonzero_right, hist_proportion_clipped)):
            # Add small gap between bars (2.5% on each side = 95% fill)
            gap = (right - left) * 0.025
            rect = Rectangle((left + gap, epsilon), right - left - 2*gap, height - epsilon,
                           facecolor='blue', edgecolor='darkblue', alpha=0.5, linewidth=1, zorder=1)
            ax.add_patch(rect)
        
        # Add dummy bar for legend
        ax.bar([0], [0], width=0, alpha=0.5, color='blue', label='Histogram', edgecolor='darkblue', linewidth=1)
        
        # Plot cumulative distribution on top
        # Clip cumulative values away from 0 and 1
        cumulative_proportion_clipped = np.clip(cumulative_proportion, epsilon, 1 - epsilon)
        ax.plot(sorted_errors, cumulative_proportion_clipped, 'r-', linewidth=3, 
                label='Cumulative', alpha=0.9, zorder=2)
        
        # Set x-axis properties
        ax.set_xlim(0, x_max)
        ax.set_xticks(x_ticks)
        ax.set_xticklabels([str(int(x)) if x >= 1 else f"{x:.1f}" for x in x_ticks])
        
        # Set y-axis properties - logit scale for both histogram and cumulative
        # Convert percentage tick values to proportions for logit scale
        # Note: logit scale works with proportions [0, 1], we display them as percentages
        y_ticks = [0.0001, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 0.9, 0.99, 0.995, 0.999, 0.9999]
        y_labels = ['0%', '0.1%', '0.5%', '1%', '5%', '10%', '50%', '90%', '99%', '99.5%', '99.9%', '100%']
        ax.set_yticks(y_ticks)
        ax.set_yticklabels(y_labels)
        ax.set_ylim(0.0001, 0.9999)  # Stay away from 0 and 1 for logit scale
        
        # Labels
        ax.set_xlabel('ULP Error', fontsize=16)
        ax.set_ylabel('Percentage (logit scale)', fontsize=16)
        
        # Title with statistics
        short_name = plot_entry.get("name", operation_name)
        title = plot_params.get("title", "ULP Error Distribution for {}")
        if title is not None:
            title = title.format(short_name)
        
        stats_text = f"Max Error: {max_error:.2f} ULP | Mean Error: {mean_error:.2f} ULP"
        ax.set_title(f"{title}\n{stats_text}", pad=20, fontsize=18)
        
        # Add legend
        ax.legend(loc='lower right', fontsize=12, framealpha=0.9)
        
        # Grid
        ax.grid(True, alpha=0.3)
        
        # Adjust layout
        plt.tight_layout()
        
        # Save the plot to all specified output paths
        for output_path in output_paths:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            plt.savefig(output_path, dpi=300, bbox_inches="tight")
            print(f"Saved histogram: {output_path}")
        
        plt.close()
        
    except Exception as e:
        import traceback
        print(f"Error plotting histogram for {plot_entry.get('id', 'unknown')}: {e}")
        traceback.print_exc()
        plt.close()


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
