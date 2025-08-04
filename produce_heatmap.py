import json
import numpy as np
import re
import glob
import os
import argparse
from collections import defaultdict

def collect_heatmap_data(problem_type, dim_range=None):
    """
    Scans for JSON files and aggregates data for the heatmap plots.

    Args:
        problem_type (str): The problem type (e.g., 'maxcut').
        dim_range (list or None): A list [min_dim, max_dim] to filter by.

    Returns:
        dict: A dictionary containing the aggregated data.
              Structure: {dim: {rank: {'primal': val, 'dual': val, ...}}}
    """
    search_path = 'results/'
    file_pattern = os.path.join(search_path, f'{problem_type}_*_trackmem_True_seeds_*_ranks_*.json')
    files = glob.glob(file_pattern)

    # Structure: {dim: {rank: {'primal': val, 'dual': val, 'dualslack': val}}}
    plot_data = defaultdict(dict)

    dim_pattern = re.compile(rf'{problem_type}_(\d+)_')
    rank_pattern = re.compile(r'_ranks_(\d+)\.json')

    print(f"Found {len(files)} files for problem type '{problem_type}'.")
    if dim_range:
        print(f"Filtering for dimensions between {dim_range[0]} and {dim_range[1]} (inclusive).")

    for file_path in files:
        basename = os.path.basename(file_path)
        dim_match = dim_pattern.search(basename)
        rank_match = rank_pattern.search(basename)

        if not dim_match or not rank_match:
            continue

        dim = int(dim_match.group(1))
        rank = int(rank_match.group(1))

        if dim_range and not (dim_range[0] <= dim <= dim_range[1]):
            continue

        try:
            with open(file_path, 'r') as f:
                data = json.load(f)

            # Calculate the max of the average for each rank type
            def get_max_avg_rank(rank_data):
                if not rank_data or not rank_data[0]:
                    return 0
                # data is nested in a list, so we take the first element
                runs = rank_data[0]
                avg_ranks = [np.mean(run) for run in runs if run]
                return np.max(avg_ranks) if avg_ranks else 0

            primal_val = get_max_avg_rank(data.get("ranksX", []))
            dual_val = get_max_avg_rank(data.get("ranksY", []))
            dualslack_val = get_max_avg_rank(data.get("ranksZ", []))
            
            plot_data[dim][rank] = {
                'primal': primal_val,
                'dual': dual_val,
                'dualslack': dualslack_val
            }

        except (json.JSONDecodeError, IndexError) as e:
            print(f"Warning: Could not process file {basename}. Error: {e}")
    
    return plot_data

def generate_dat_file_content(plot_data):
    """
    Generates and prints the content for the .dat files, with a zero-padded border,
    and recommends a color scale range.
    """
    if not plot_data:
        print("No data available to generate .dat files.")
        return

    # Get the actual dimensions and ranks found in the data
    all_dims = sorted(plot_data.keys())
    all_ranks = sorted(list(set(rank for dim_data in plot_data.values() for rank in dim_data.keys())))
    
    # The grid size will be one larger than the number of dims/ranks to accommodate the zero-padding
    grid_width = len(all_ranks) + 1
    grid_height = len(all_dims) + 1

    # --- Calculate global min/max for color scale recommendation ---
    all_values = []
    for dim_data in plot_data.values():
        for rank_data in dim_data.values():
            for key in ['primal', 'dual', 'dualslack']:
                value = rank_data.get(key, 0)
                if value > 0:
                    all_values.append(value)
    
    recommendation = "No non-zero data found to recommend a color scale."
    if all_values:
        point_meta_min = np.floor(min(all_values))
        point_meta_max = np.ceil(max(all_values))
        recommendation = f"point meta min={int(point_meta_min)}, point meta max={int(point_meta_max)},"


    # --- Helper to generate content for one file ---
    def generate_single_file(file_name, data_key):
        print(f"\\begin{{filecontents*}}{{{file_name}}}")
        print("x y val")
        # Loop through the grid coordinates
        for y in range(grid_height):
            for x in range(grid_width):
                value = 0.0 # Default to 0 for the padded border
                # If not on the border, get the actual data
                if x > 0 and y > 0:
                    # Map grid coordinates back to actual dim and rank
                    # y-1 and x-1 because grid is 1-indexed for data
                    dim = all_dims[y - 1]
                    rank = all_ranks[x - 1]
                    value = plot_data.get(dim, {}).get(rank, {}).get(data_key, 0)
                
                print(f"{x} {y} {value:.4f}")
        print("\\end{filecontents*}")

    # --- Generate all three files ---
    generate_single_file("primal.dat", "primal")
    print("\n")
    generate_single_file("dual.dat", "dual")
    print("\n")
    generate_single_file("dualslack.dat", "dualslack")
    
    # --- Print the recommendation ---
    print("\n" + "-"*20)
    print("Recommended color scale setting:")
    print(recommendation)
    print("-" * 20)


# --- Main Execution ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate .dat file content for LaTeX heatmaps from experiment data.")
    parser.add_argument(
        "problem_type", 
        choices=['maxcut', 'max_stable_set', 'graphm'],
        help="The type of problem to process (e.g., 'maxcut')."
    )
    parser.add_argument(
        "--dims",
        nargs=2,
        type=int,
        metavar=('MIN_DIM', 'MAX_DIM'),
        help="The inclusive range of dimensions to scan (e.g., --dims 6 9)."
    )
    args = parser.parse_args()

    # 1. Collect and process data
    data = collect_heatmap_data(args.problem_type, args.dims)

    # 2. Generate and print the .dat file contents
    if data:
        print("\n" + "-"*20)
        print("Copy the LaTeX filecontents below:")
        print("-" * 20 + "\n")
        generate_dat_file_content(data)