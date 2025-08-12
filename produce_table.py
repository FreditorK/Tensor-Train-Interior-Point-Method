import json
import numpy as np
import re
import glob
import os
import argparse

def format_scientific(mean_val, std_val):
    """
    Formats the mean and standard deviation into a LaTeX-compatible
    scientific notation string: (mean \\pm std) \\times 10^{exponent}.
    """
    if mean_val == 0 and std_val == 0:
        return "$0.00 \\pm 0.00$"
        
    if mean_val == 0:
        # If mean is zero, base the exponent on the standard deviation
        exponent = int(np.floor(np.log10(abs(std_val)))) if std_val != 0 else 0
        mean_mantissa = 0
    else:
        exponent = int(np.floor(np.log10(abs(mean_val))))
        mean_mantissa = mean_val / 10**exponent

    if std_val == 0:
        std_mantissa = 0
    else:
        std_mantissa = std_val / 10**exponent

    # Format to 2 decimal places
    mean_str = f"{mean_mantissa:.2f}"
    std_str = f"{std_mantissa:.2f}"

    if exponent == 0:
        return f"${mean_str} \\pm {std_str}$"
    else:
        return f"$({mean_str} \\pm {std_str}) \\times 10^{{{exponent}}}$"

def process_json_to_latex(file_path, problem_type, method_name="TT-IPM", total_rows=1):
    """
    Reads a JSON file containing experiment results, calculates statistics,
    and formats them into a LaTeX table row.

    Args:
        file_path (str): The path to the input JSON file.
        problem_type (str): The type of problem (e.g., 'maxcut').
        method_name (str): The name of the method to display in the table.
                           Set to None to omit it (for subsequent rows).
        total_rows (int): The total number of rows for the \multirow command.
    """
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"Error: The file '{file_path}' was not found.")
        return None
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from the file '{file_path}'.")
        return None

    # --- Data Extraction ---
    memory_mb = np.array(data.get("memory", [[]])[0])
    runtimes_s = np.array(data.get("runtimes", [[]])[0])
    primal_errors = np.array(data.get("feasibility_errors", [[]])[0])
    dual_errors = np.array(data.get("dual_feasibility_errors", [[]])[0])
    duality_gaps = np.array(data.get("complementary_slackness", [[]])[0])
    iterations = np.array(data.get("num_iters", [[]])[0])

    # --- Dimension Extraction ---
    # Extract 'dim' from the config string first
    config_str = data.get("config_str", "")
    match_config = re.search(r"'dim':\s*(\d+)", config_str)
    if match_config:
        dim = int(match_config.group(1))
    else:
        # Fallback to filename if 'dim' is not in config
        pattern = re.compile(rf'{problem_type}_(\d+)_')
        match_fn = pattern.search(os.path.basename(file_path))
        if not match_fn:
            print(f"Error: Could not extract 'dim' from config or filename for {file_path}.")
            return None
        dim = int(match_fn.group(1))

    # --- Statistics Calculation ---
    mem_mean, mem_std = np.mean(memory_mb), np.std(memory_mb)
    run_mean, run_std = np.mean(runtimes_s), np.std(runtimes_s)
    p_err_mean, p_err_std = np.mean(primal_errors), np.std(primal_errors)
    d_err_mean, d_err_std = np.mean(dual_errors), np.std(dual_errors)
    gap_mean, gap_std = np.mean(duality_gaps), np.std(duality_gaps)
    iter_mean, iter_std = np.mean(iterations), np.std(iterations)

    # --- LaTeX String Formatting ---
    method_cell = f"\\multirow{{{total_rows}}}{{*}}{{{method_name}}}\n" if method_name else ""
    storage_str = f"${mem_mean:.2f} \\pm {mem_std:.2f}$ MB"
    runtime_str = f"${run_mean:.2f} \\pm {run_std:.2f}$s"
    primal_err_str = format_scientific(p_err_mean, p_err_std)
    dual_err_str = format_scientific(d_err_mean, d_err_std)
    duality_gap_str = format_scientific(gap_mean, gap_std)
    iterations_str = f"${iter_mean:.1f} \\pm {iter_std:.1f}$"
    ex = 2*dim if problem_type == "graphm" else dim
    size_str = f"$2^{{{ex}}}$"

    latex_row = (
        f"{method_cell} & {storage_str} & {runtime_str} & {primal_err_str} & "
        f"{dual_err_str} & {duality_gap_str} & {iterations_str} & {size_str} \\\\"
    )
    return latex_row


# --- Main Execution ---
if __name__ == "__main__":
    # --- Command-Line Argument Parsing ---
    parser = argparse.ArgumentParser(description="Generate a LaTeX table from experiment result JSON files.")
    parser.add_argument(
        "problem_type", 
        choices=['maxcut', 'max_stable_set', 'graphm', 'corr_clust'],
        help="The type of problem to process (e.g., 'maxcut')."
    )
    args = parser.parse_args()
    problem_type = args.problem_type

    # --- File Discovery and Sorting ---
    search_path = 'results/'
    file_pattern = os.path.join(search_path, f'{problem_type}_*_trackmem_True_seeds_*_ranks_1.json')
    files = glob.glob(file_pattern)

    def get_dim_from_filename(filename, p_type):
        match = re.search(rf'{p_type}_(\d+)_', os.path.basename(filename))
        return int(match.group(1)) if match else 0
    
    files.sort(key=lambda f: get_dim_from_filename(f, problem_type))

    # --- Table Generation ---
    if not files:
        print(f"No files found in '{search_path}' for problem type '{problem_type}'.")
        print(f"Searched with pattern: {file_pattern}")
    else:
        print(f"Found and processing {len(files)} files for '{problem_type}':")
        for f in files:
            print(f"- {os.path.basename(f)}")
        print("-" * 20)
        print("Copy the LaTeX code below:")
        print("-" * 20)

        print("Method & Storage & Runtime & Primal Error & Dual Error & Duality Gap & Iterations & Size \\\\")
        print("\\midrule")

        num_files = len(files)
        for i, json_file in enumerate(files):
            if i == 0:
                row = process_json_to_latex(json_file, problem_type, method_name="TT-IPM", total_rows=num_files)
            else:
                row = process_json_to_latex(json_file, problem_type, method_name=None)
            
            if row:
                print(row)