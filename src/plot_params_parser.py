import json
import os
import multiprocessing as mp
import pandas as pd
import numpy as np


RED = "\033[91m"
GREEN = "\033[92m"
BLUE = "\033[94m"
YELLOW = "\033[93m"
RESET = "\033[0m"


def load_csv(filename):
    return pd.read_csv(filename, sep=",", index_col=False, skipinitialspace=True)


def create_directory(file_path):
    dir_path = os.path.dirname(file_path)
    os.makedirs(dir_path, exist_ok=True)


def parse_plot_config(plot_config_path):
    with open(plot_config_path, "r") as f:
        plot_config = json.load(f)

    for group in plot_config["groups"]:
        for plot_entry in group["plots"]:
            insert_default_params(group, plot_entry)

    return plot_config


def hash_plot_entry(entry):
    import hashlib

    entry_str = json.dumps(entry, sort_keys=True)
    hash_obj = hashlib.sha256(entry_str.encode())
    return hash_obj.hexdigest()


def generate_plot_config_hashes(plot_config):
    # Dictionary to store hashes for each plot configuration
    plot_config_hashes = {}

    # Process each plot entry in the configuration
    for group in plot_config["groups"]:
        for plot_entry in group["plots"]:
            if "data" in plot_entry:
                del plot_entry["data"]

            hash_value = hash_plot_entry(plot_entry)
            plot_id = plot_entry["id"]

            # Store the hash with its corresponding plot entry
            plot_config_hashes[plot_id] = hash_value

    return plot_config_hashes


def save_plot_config_hashes(plot_config_hashes, output_path):
    # Save as json file
    with open(output_path, "w") as f:
        json.dump(plot_config_hashes, f)


def load_plot_config_hashes(input_path):
    if not os.path.exists(input_path):
        plot_config_hashes = {}
        plot_config_hashes["last_modified"] = 0
        return plot_config_hashes

    # Get time stamp of last modification of hash file
    hash_last_modified = os.path.getmtime(input_path)

    # Load as json file
    with open(input_path, "r") as f:
        plot_config_hashes = json.load(f)

    plot_config_hashes["last_modified"] = hash_last_modified

    return plot_config_hashes


# For a given plot, insert default parameters if missing from plots
def insert_default_params(plot_parent_config, plot_group_config):
    for default_param in plot_parent_config["default_params"]:
        if default_param not in plot_group_config:
            plot_group_config[default_param] = plot_parent_config["default_params"][default_param]

    if "default_params" in plot_parent_config:
        default_plot_params = plot_parent_config["default_params"]["plot_params"]
        for default_plot_param in default_plot_params:
            if default_plot_param not in plot_group_config["plot_params"]:
                plot_group_config["plot_params"][default_plot_param] = default_plot_params[default_plot_param]


def plot_all(
    plot_fun, plot_config, base_output_dir, last_hashes, current_hashes, force_replot=False, sanitize_fun=None
):
    plot_args = []

    for plot_group in plot_config["groups"]:
        for plot_entry in plot_group["plots"]:
            plot_id = plot_entry["id"]

            do_replot = True

            if plot_id in current_hashes:
                # Compute new hash and check against previous one
                hash_value = current_hashes[plot_id]
                last_hash = last_hashes[plot_id] if plot_id in last_hashes else None

                if hash_value == last_hash:
                    print(f"{BLUE}Skipping {plot_id} because hash value is the same{RESET}")
                    do_replot = False

            last_modified = last_hashes["last_modified"]
            entry_present = False
            for file in plot_entry["files"]:
                if not os.path.exists(file):
                    print(f"{RED}Skipping file {file}: measurements not found{RESET}")
                    continue
                else:
                    entry_present = True

                if os.path.getmtime(file) > last_modified:
                    do_replot = True
                    break

            if not entry_present:
                continue

            if force_replot:
                do_replot = True

            if do_replot:
                plot_args.append(plot_entry)

    # For each plot entry, import data
    for plot_entry in plot_args:
        data_series = plot_entry["files"]
        list_all_data = []

        # TODO: Cache data
        for series in data_series:
            if not os.path.exists(series):
                continue

            data_op = load_csv(series)
            list_all_data.append(data_op)

        data = pd.concat(list_all_data, axis=0)

        if sanitize_fun is not None:
            data = sanitize_fun(data)

        # data = data.set_index(["base_x", "operation"])

        all_outputs = plot_entry["outputs"]
        for output_path in all_outputs:
            create_directory(output_path)

        # Transform data if necessary
        plot_entry["data"] = data

    # Launch parallel plots
    num_processes = mp.cpu_count()
    print(f"Plotting {len(plot_args)} operations with {num_processes} processes")

    with mp.Pool(num_processes) as pool:
        results = pool.map(plot_fun, plot_args)

        cnt = 1
        for result in results:
            print(f"#{cnt}/{len(results)}", end="\r")
            cnt += 1
