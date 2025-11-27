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
        levels = [1e-6] + [item["threshold"] for item in colormap_config]
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

    last_hashes = load_plot_config_hashes(f"accuracy_results/binary-plot-hashes.csv")

    # Get time stamp of last modification of this script
    script_mtime = os.path.getmtime(__file__)
    force_replot = False
    if script_mtime > last_hashes["last_modified"]:
        force_replot = True

    current_hashes = generate_plot_config_hashes(plot_config)

    plot_all(plot_heatmap, plot_config, dest_dir, last_hashes, current_hashes, force_replot, sanitize_data)

    save_plot_config_hashes(current_hashes, f"accuracy_results/binary-plot-hashes.csv")

    print("All JSON-configured heatmaps generated successfully!")


if __name__ == "__main__":
    main()
