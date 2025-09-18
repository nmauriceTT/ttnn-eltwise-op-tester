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


def create_heatmap(data, operation_name, output_path):
    """
    Create a heatmap from binary operation accuracy data

    Args:
        data: DataFrame with columns 'a', 'b', 'max_ulp_error'
        operation_name: Name of the operation for the title
        output_path: Path where to save the PNG file
    """
    # Create pivot table with a on x-axis, b on y-axis, max_ulp_error as values

    print(f"DATA = \n{data}")

    pivot_data = data.pivot(columns="a", index="b", values="max_ulp_error")

    print(f"COLUMNS = {pivot_data.columns.tolist()}")
    print(f"INDEX = {pivot_data.index.tolist()}")
    # print(f"VALUES = {pivot_data.values.tolist()}")

    # Ensure the pivot table has the right dimensions (512x512)
    print(f"Pivot table shape: {pivot_data.shape}")

    # Create figure with appropriate size
    plt.figure(figsize=(12, 10))

    # Create custom colormap
    # Colors: good (green) for <2 ULP, bad (red) for >5 ULP
    # Black for NaN values
    colors = [
        "#2E8B57",  # Sea Green (good, <2 ULP)
        "#FFD700",  # Gold (moderate, 2-5 ULP)
        "#FF6347",  # Tomato (bad, >5 ULP)
        "#8B0000",
    ]  # Dark Red (very bad, >10 ULP)

    # Define color boundaries
    boundaries = [0, 2, 5, 10, 1e9]

    # Create custom colormap
    cmap = mcolors.ListedColormap(colors)
    norm = mcolors.BoundaryNorm(boundaries, cmap.N)

    # Handle NaN values by setting them to a specific color (black)
    # First, create a mask for NaN values
    # pivot_data = pivot_data.fillna(1e9)
    # pivot_data = pivot_data.replace([np.inf, -np.inf], 1e9)

    nan_mask = np.isnan(pivot_data.values)

    print(f"NAN MASK = {nan_mask}")
    # assert np.all(nan_mask == False)

    print(f"Pivot data shape: {pivot_data.shape}")
    print(f"Pivot data: {pivot_data}")

    # Create the color mesh
    # Get the x and y coordinates for pcolormesh
    x_coords = pivot_data.columns.values
    y_coords = pivot_data.index.values

    # Create meshgrid for pcolormesh
    X, Y = np.meshgrid(x_coords, y_coords)

    levels = [1e-6, 1, 2, 3, 4, 5, 10, 100, 100]
    cmap = plt.colormaps["PiYG"]
    cmap.set_bad(color="purple")

    norm = BoundaryNorm(levels, ncolors=cmap.N, clip=True)

    # norm=LogNorm(vmin=1e-1, vmax=100)
    # Create the color mesh
    mesh = plt.pcolormesh(X, Y, pivot_data.values, norm=norm, cmap="PuBu_r", shading="auto")

    # Create colorbar
    cbar = plt.colorbar(mesh, norm=norm, label="Max ULP Error", ticks=levels)

    # Set background color for NaN values to Yellow
    plt.gca().set_facecolor("yellow")

    plt.gca().set_xscale("log")
    plt.gca().set_yscale("symlog")

    plt.gca().set_xlim(1e-12, 1e12)
    plt.gca().set_ylim(-1e3, 1e3)

    # Add a horizontal red line at B == 0 (y == 0)
    plt.axhline(y=0, color="red", linestyle="--", linewidth=1.5, label="y = a**0", alpha=0.5)
    plt.text(x=plt.gca().get_xlim()[0], y=0, s="$y = a^{0}$", color="red", va="bottom", ha="left")

    # Add a vertical red line at A == 1 (x == 1)
    plt.axvline(x=1, color="red", linestyle="--", linewidth=1.5, label="y = 1**b", alpha=0.5)
    plt.text(x=2, y=plt.gca().get_ylim()[1] * 0.9, s="$y = 1^{b}$", color="red", va="top", ha="left")

    # Set aspect ratio to be square
    # plt.gca().set_aspect('equal')

    # Set title and labels
    plt.title(f"{operation_name}(A, B) - ULP Error Heatmap\n(Yellow = NaN or Inf)", pad=20)
    plt.xlabel("A (base)")
    plt.ylabel("B (exponent)")

    # Adjust layout to prevent label cutoff
    plt.tight_layout()

    # Save the plot
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    plt.savefig(f"{output_path}.png", dpi=300, bbox_inches="tight")
    # plt.savefig(f"{output_path}.svg", bbox_inches="tight")
    plt.close()


def plot_heatmap(plot_entry):
    """
    Create a heatmap from binary operation accuracy data using parameters from JSON config.
    Similar to the plot function in plot_accuracy.py but adapted for heatmaps.

    Args:
        plot_entry: Dictionary containing plot configuration from binary-plot-params.json
    """
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
    levels = [1e-6, 1, 2, 3, 4, 5, 10, 100, 100]
    cmap = plt.colormaps["PiYG"]
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

    plt.gca().set_xscale("symlog")
    plt.gca().set_yscale("symlog")

    # Use limits from JSON config
    plt.gca().set_xlim(alim[0], alim[1])
    plt.gca().set_ylim(blim[0], blim[1])

    # Add reference lines
    plt.axhline(y=0, color="red", linestyle="--", linewidth=1.5, label="y = a**0", alpha=0.5)
    plt.text(x=plt.gca().get_xlim()[0], y=0, s="$y = a^{0}$", color="red", va="bottom", ha="left")

    plt.axvline(x=1, color="red", linestyle="--", linewidth=1.5, label="y = 1**b", alpha=0.5)
    plt.text(x=2, y=plt.gca().get_ylim()[1] * 0.9, s="$y = 1^{b}$", color="red", va="top", ha="left")

    # Set title and labels
    plt.title(f"{operation_name}(A, B) - ULP Error Heatmap\n(Yellow = NaN or Inf)", pad=20)
    plt.xlabel("X")
    plt.ylabel("Y")

    # Adjust layout to prevent label cutoff
    plt.tight_layout()

    # Save the plot to all specified output paths
    for output_path in output_paths:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches="tight")

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


def old_main():
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
    print("Generating heatmaps using original create_heatmap function...")

    csv_file = "accuracy_results/results/binary/divide[bfloat16].csv"
    data = load_csv(csv_file)
    data = preprocess_data(data)
    create_heatmap(data, "divide", f"{output_dir}/divide")

    csv_file = "accuracy_results/results/binary/div[bfloat16].csv"
    data = load_csv(csv_file)
    data = preprocess_data(data)
    create_heatmap(data, "div", f"{output_dir}/div")

    csv_file = "accuracy_results/results/binary/div-accurate[bfloat16].csv"
    data = load_csv(csv_file)
    data = preprocess_data(data)
    create_heatmap(data, "div-accurate", f"{output_dir}/div-accurate")

    print("All heatmaps generated successfully!")

    # Example 2: Using the new plot_heatmap function with JSON configuration
    print("\nGenerating heatmaps using plot_heatmap function with JSON config...")

    # Load configuration from JSON file
    config_path = "eltwise-accuracy/binary-plot-params.json"
    plot_configs = parse_binary_plot_config(config_path)

    # Process each plot configuration
    for plot_config in plot_configs:
        # Load data for this plot
        csv_file = plot_config["files"][0]  # Assuming single file per plot
        if os.path.exists(csv_file):
            data = load_csv(csv_file)
            data = preprocess_data(data)

            # Add data to plot_config
            plot_config["data"] = data

            # Generate heatmap using plot_heatmap function
            plot_heatmap(plot_config)
            print(f"Generated heatmap for {plot_config['name']}")
        else:
            print(f"Warning: File {csv_file} not found, skipping {plot_config['name']}")

    print("All JSON-configured heatmaps generated successfully!")


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

    plot_config = parse_plot_config("eltwise-accuracy/binary-plot-params.json")

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
