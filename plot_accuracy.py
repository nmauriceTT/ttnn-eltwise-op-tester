import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.patches as mpatches
from matplotlib.ticker import ScalarFormatter, PercentFormatter


import pandas as pd
import seaborn as sns
import numpy as np

import json
import math
import sys
import os.path
import re
import time
import multiprocessing as mp

from plot_params_parser import (
    parse_plot_config,
    generate_plot_config_hashes,
    save_plot_config_hashes,
    load_plot_config_hashes,
    plot_all,
)

RED = "\033[91m"
GREEN = "\033[92m"
BLUE = "\033[94m"
YELLOW = "\033[93m"
RESET = "\033[0m"


plt.rcParams["svg.fonttype"] = "none"  # Make text editing in SVG easier


def preprocess_operation(data):
    # Somewhat hacky way to have differenciate series on the same legend
    data["operation_mean"] = data["operation"] + " (mean)"
    data["operation_max"] = data["operation"] + " (max)"

    return data


def try_plot(plot_entry):
    try:
        plot(plot_entry)
    except Exception as e:
        print(f"Error plotting {RED}{plot_entry['id']}: {e}{RESET}")


# Remove data where inputs are subnormals
def remove_subnormals(data, min_normal_value=2**-126):
    data = data[data["base_x"].abs() >= min_normal_value]
    data = data[data["base_y"].abs() >= min_normal_value]
    return data


def plot(plot_entry):
    all_outputs = plot_entry["outputs"]
    data = plot_entry["data"]

    # Read parameters


    plot_params = plot_entry["plot_params"]

    id = plot_entry["id"]


    title = plot_params["title"] if "title" in plot_params else None
    short_name = plot_entry["name"] if "name" in plot_entry else id


    if title is not None:
        title = title.format(short_name)

    xbase = plot_params["xbase"] if "xbase" in plot_params else 10
    xscale = plot_params["xscale"] if "xscale" in plot_params else "symlog"

    yscale = plot_params["yscale"] if "yscale" in plot_params else "asinh"
    ybase = plot_params["ybase"] if "ybase" in plot_params else 10

    [xmin, xmax] = plot_params["xlim"] if "xlim" in plot_params else [None, None]
    [ymin, ymax] = plot_params["ylim"] if "ylim" in plot_params else [None, None]

    xticks = plot_params["xticks"] if "xticks" in plot_params else None
    yticks = plot_params["yticks"] if "yticks" in plot_params else None

    palette_offset = plot_params["palette_offset"] if "palette_offset" in plot_params else 0

    xname = plot_entry["xname"]
    ynames = plot_entry["ynames"]
    hseries = plot_entry["hue"] if "hue" in plot_entry else None

    xlabel = plot_params["xlabel"] if "xlabel" in plot_params else xname
    ylabel = plot_params["ylabel"] if "ylabel" in plot_params else ynames[0]

    yticksformat = plot_params["yticksformat"] if "yticksformat" in plot_params else None

    plot_type = plot_entry["type"] if "type" in plot_entry else "lineplot"
    print(f"Plot type: {id} - {plot_type}")

    if "keep_subnormals" not in plot_params:
        data = remove_subnormals(data)

    # Remove data that exceeds xmin,xmax,ymin,ymax
    if xmin is not None:
        data = data[data[xname] >= xmin]
    if xmax is not None:
        data = data[data[xname] <= xmax]

    fig, ax = plt.subplots(figsize=(25, 15))

    # color_palette = sns.color_palette("deep", len(ynames))
    ncolors = len(data[hseries].unique())
    color_palette = sns.color_palette("deep", ncolors + palette_offset)[palette_offset:]


    for y in ynames:
        d2 = data.copy()

        [yname, ysuffix, linestyle] = y
        d2["operation"] += " " + ysuffix

        # Remove data that exceeds ymin,ymax
        # if ymin is not None:
        #     d2 = d2[d2[yname] >= ymin]
        # if ymax is not None:
        #     d2 = d2[d2[yname] <= ymax]


        if plot_type == "lineplot":
            if hseries is not None:
                ax = sns.lineplot(
                    data=d2, x=xname, y=yname, ax=ax, hue=hseries, linestyle=linestyle, palette=color_palette
                )
            else:
                ax = sns.lineplot(
                    data=d2, x=xname, y=yname, ax=ax, label=yname, linestyle=linestyle, palette=color_palette
                )
        elif plot_type == "scatterplot":
            ax = sns.scatterplot(data=d2, x=xname, y=yname, ax=ax, hue=hseries, palette=color_palette, edgecolor="none")

    if xscale == "linear":
        ax.set_xscale("linear")
    else:
        ax.set_xscale(xscale, base=xbase)

    if yscale == "asinh":
        ax.set_yscale(yscale, linear_width=0.01)
    elif yscale == "linear":
        ax.set_yscale("linear")
    else:
        ax.set_yscale(yscale, base=ybase)

    if title is not None:
        ax.set_title(title)

    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    if "vertical_lines" in plot_params:
        for vertical_line in plot_params["vertical_lines"]:

            # Do not plot lines if outside graphs
            if xmax is not None and vertical_line[0] > xmax:
                continue
            if xmin is not None and vertical_line[0] < xmin:
                continue

            ax.axvline(x=vertical_line[0], color="k", linestyle="--")
            label_y = ax.get_ylim()[1] / 2
            ax.text(vertical_line[0], label_y, vertical_line[1])

    if "horizontal_lines" in plot_params:
        for horizontal_line in plot_params["horizontal_lines"]:

            # Do not plot lines if outside graphs
            if ymax is not None and horizontal_line[0] > ymax:
                continue
            if ymin is not None and horizontal_line[0] < ymin:
                continue

            ax.axhline(y=horizontal_line[0], color="k", linestyle="--")
            label_x = ax.get_xlim()[1] / 2
            ax.text(label_x, horizontal_line[0], horizontal_line[1])

    if yticks is not None:
        # print(f"yticks = {yticks}")
        ax.set_yticks(yticks)

    if yticksformat == "percent":
        # plt.gca().set_yticklabels([f"{100*x}%" for x in ax.get_yticks()])
        ax.yaxis.set_major_formatter(PercentFormatter(xmax=1))

    if xticks is not None:
        ax.set_xticks(xticks)

    for output_path in all_outputs:
        plt.savefig(output_path, bbox_inches="tight", pad_inches=0.0)

    plt.close()


def sanitize_data(data):
    # Remove data with NaN or infinity (might speedup plotting)
    data = data[(data["base_x"].notna()) & (np.isfinite(data["base_x"]))]
    data = data.reset_index()

    return data


def main():
    sns.set(
        style="ticks",
        rc={
            "axes.grid": True,
            # "axes.edgecolor": None,
            "axes.titlesize": 60,
            "axes.labelsize": 60,
            "xtick.labelsize": 60,
            "ytick.labelsize": 60,
            "font.size": 60,
            "legend.title_fontsize": 50,
            "legend.fontsize": 40,
            "lines.linewidth": 4,
            "axes.linewidth": 1,
            "font.serif": ["Latin Modern Math"],
            "lines.markersize": 8,
            "lines.markeredgecolor": "none",
        },
    )

    accuracy_dir = "accuracy_results/results/"
    dest_dir = "accuracy_results/plots"

    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)

    if not os.path.exists(f"{dest_dir}/abs/"):
        os.makedirs(f"{dest_dir}/abs/")

    plot_config = parse_plot_config("eltwise-accuracy/plot-params.json")

    last_hashes = load_plot_config_hashes(f"accuracy_results/plot-hashes.csv")

    # Get time stamp of last modification of this script
    script_mtime = os.path.getmtime(__file__)
    force_replot = False
    if script_mtime > last_hashes["last_modified"]:
        force_replot = True

    current_hashes = generate_plot_config_hashes(plot_config)

    plot_all(try_plot, plot_config, dest_dir, last_hashes, current_hashes, force_replot, sanitize_data)

    save_plot_config_hashes(current_hashes, f"accuracy_results/plot-hashes.csv")


if __name__ == "__main__":
    main()
