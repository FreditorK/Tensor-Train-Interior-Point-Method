import json
import numpy as np
import re
import glob
import os
import argparse
from collections import defaultdict

def collect_raw_plot_data(problem_type, dim_range=None):
    """
    Scans for JSON files and aggregates raw runtime and memory data,
    grouped by dimension, filtering by a specified dimension range.

    Args:
        problem_type (str): The problem type (e.g., 'maxcut').
        dim_range (list or None): A list [min_dim, max_dim] to filter by.
                                  If None, all dimensions are included.

    Returns:
        dict: A dictionary containing the raw plot data.
              Structure: {dim: {'runtime_points': [(r1, t1), ...], ...}}
    """
    search_path = 'results/'
    # Pattern to find files with any dim and any rank for the given problem type
    file_pattern = os.path.join(search_path, f'{problem_type}_*_trackmem_True_seeds_*_ranks_*.json')
    files = glob.glob(file_pattern)

    # Use defaultdict to simplify data aggregation
    plot_data = defaultdict(lambda: {'runtime_points': [], 'memory_points': []})

    # Regex to extract dim and rank from filenames
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
            print(f"Warning: Could not parse dim/rank from filename: {basename}")
            continue

        dim = int(dim_match.group(1))
        rank = int(rank_match.group(1))

        # Filter by dimension range if one is provided
        if dim_range and not (dim_range[0] <= dim <= dim_range[1]):
            continue

        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
            
            raw_runtimes = data.get("runtimes", [[]])[0]
            raw_memories = data.get("memory", [[]])[0]

            for rt in raw_runtimes:
                plot_data[dim]['runtime_points'].append((rank, rt))
            
            for mem in raw_memories:
                plot_data[dim]['memory_points'].append((rank, mem))

        except (json.JSONDecodeError, IndexError) as e:
            print(f"Warning: Could not process file {basename}. Error: {e}")
    
    return plot_data

def generate_addplot_lines(plot_data):
    """
    Generates and prints the LaTeX \\addplot lines for scatter plots
    using symbolic x-coordinates with a horizontal jitter to avoid overlap.
    """
    if not plot_data:
        print("No data available to generate plot lines.")
        return

    colors = ['blue', 'red', 'green!60!black', 'orange', 'purple', 'brown']
    sorted_dims = sorted(plot_data.keys())
    num_dims = len(sorted_dims)
    
    jitter_step = 6 

    # --- Generate Runtime Plot Lines ---
    print("% Runtime Plot Lines")
    for i, dim in enumerate(sorted_dims):
        color = colors[i % len(colors)]
        xshift = (i - (num_dims - 1) / 2.0) * jitter_step
        
        runtime_coords = " ".join([
            f"({r}, {rt:.2f})" 
            for r, rt in plot_data[dim]['runtime_points']
        ])
        
        if runtime_coords:
            # Plot the actual data points but hide them from the legend
            plot_options = (
                f"only marks, color={color}, mark=*, mark size=2pt, "
                f"opacity=0.7, draw opacity=0, xshift={xshift}pt, forget plot"
            )
            print(f"\\addplot+[{plot_options}] coordinates {{{runtime_coords}}};")
            # Add a dummy plot just for the legend entry with a square symbol
            print(f"\\addlegendimage{{only marks, mark=square*, mark options={{fill={color}, draw=none}}, color={color}}}")

    
    print("\n% Memory Plot Lines")
    # --- Generate Memory Plot Lines ---
    for i, dim in enumerate(sorted_dims):
        color = colors[i % len(colors)]
        xshift = (i - (num_dims - 1) / 2.0) * jitter_step

        memory_coords = " ".join([
            f"({r}, {mem:.2f})" 
            for r, mem in plot_data[dim]['memory_points']
        ])

        if memory_coords:
            # Plot the actual data points but hide them from the legend
            plot_options = (
                f"only marks, color={color}, mark=*, mark size=2pt, "
                f"opacity=0.7, draw opacity=0, xshift={xshift}pt, forget plot"
            )
            print(f"\\addplot+[{plot_options}] coordinates {{{memory_coords}}};")
            # Add a dummy plot just for the legend entry with a square symbol
            print(f"\\addlegendimage{{only marks, mark=square*, mark options={{fill={color}, draw=none}}, color={color}}}")


# --- Main Execution ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate LaTeX \\addplot lines for scatter plots from experiment data.")
    parser.add_argument(
        "problem_type", 
        choices=['maxcut', 'max_stable_set', 'graphm', 'corr_clust'],
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

    # 1. Collect raw data points, passing the dimension range
    raw_data = collect_raw_plot_data(args.problem_type, args.dims)

    # 2. Generate and print the LaTeX plot lines
    if raw_data:
        print("\n" + "-"*20)
        print("Copy the LaTeX plot lines below:")
        print("-" * 20 + "\n")
        generate_addplot_lines(raw_data)
        print("\n" + "-" * 20)
