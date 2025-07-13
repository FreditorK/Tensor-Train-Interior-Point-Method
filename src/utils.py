import numpy as np
import argparse
import os
import sys
import time
import yaml
from memory_profiler import memory_usage

sys.path.append(os.getcwd() + '/../../')

from src.tt_ops import *
from src.tt_ipm import tt_ipm

def run_experiment(create_problem_fn):

    np.set_printoptions(linewidth=np.inf, threshold=np.inf, precision=4, suppress=True)

    parser = argparse.ArgumentParser(description="Script with optional memory tracking.")
    parser.add_argument("--track_mem", action="store_true", help="Enable memory tracking from a certain point.")
    parser.add_argument("--config", type=str, required=True, help="Path to the YAML configuration file")
    args = parser.parse_args()

    with open(os.path.join(os.getcwd(), "../../", args.config), "r") as file:
        config = yaml.safe_load(file)

    num_ranks = len(config["max_ranks"])
    num_seeds = len(config["seeds"])
    dim = config["dim"]

    # Always allocate the common arrays
    problem_creation_times = np.zeros((num_ranks, num_seeds))
    runtimes = np.zeros((num_ranks, num_seeds))
    memory = np.zeros((num_ranks, num_seeds))
    complementary_slackness = np.zeros((num_ranks, num_seeds))
    feasibility_errors = np.zeros((num_ranks, num_seeds))
    dual_feasibility_errors = np.zeros((num_ranks, num_seeds))
    num_iters = np.zeros((num_ranks, num_seeds))

    if "graphm" in args.config:
        ranksX = np.zeros((num_ranks, num_seeds, 2 * dim))
        ranksY = np.zeros((num_ranks, num_seeds, 2 * dim))
        ranksZ = np.zeros((num_ranks, num_seeds, 2 * dim))
        ranksT = np.zeros((num_ranks, num_seeds, 2 * dim))
    else:
        ranksX = np.zeros((num_ranks, num_seeds, dim - 1))
        ranksY = np.zeros((num_ranks, num_seeds, dim - 1))
        ranksZ = np.zeros((num_ranks, num_seeds, dim - 1))
        ranksT = None

    for r_i, rank in enumerate(config["max_ranks"]):
        print(f"\n===== Processing Rank: {rank} =====")
        for s_i, seed in enumerate(config["seeds"]):
            print(f"  --- Running Seed: {seed} ---")
            np.random.seed(seed)
            t1 = time.time()

            # Support either 4-return or 5-return variant
            problem = create_problem_fn(dim, rank)

            if len(problem) == 5:
                obj_tt, L_op_tt, bias_tt, ineq_mask, lag_maps = problem
            else:
                obj_tt, L_op_tt, bias_tt, lag_y = problem
                ineq_mask = None
                lag_maps = {"y": lag_y}

            # Reshape everything as needed
            lag_maps = {k: tt_reshape(v, (4, 4)) for k, v in lag_maps.items()}
            obj_tt = tt_reshape(obj_tt, (4,))
            bias_tt = tt_reshape(bias_tt, (4,))
            t2 = time.time()

            def run_ipm():
                return tt_ipm(
                    lag_maps,
                    obj_tt,
                    L_op_tt,
                    bias_tt,
                    ineq_mask=ineq_mask,
                    max_iter=config["max_iter"],
                    verbose=config["verbose"],
                    gap_tol=config["gap_tol"],
                    op_tol=config["op_tol"],
                    warm_up=config["warm_up"],
                    aho_direction=False,
                    mals_restarts=config["mals_restarts"],
                    max_refinement=config["max_refinement"]
                )

            if args.track_mem:
                start_mem = memory_usage(max_usage=True, include_children=True)

                def wrapper(): return run_ipm()

                res = memory_usage(proc=wrapper, max_usage=True, retval=True, include_children=True)
                X_tt, Y_tt, T_tt, Z_tt, info = res[1]
                memory[r_i, s_i] = res[0] - start_mem
            else:
                X_tt, Y_tt, T_tt, Z_tt, info = run_ipm()

            t3 = time.time()

            # Store metrics
            problem_creation_times[r_i, s_i] = t2 - t1
            runtimes[r_i, s_i] = t3 - t2
            complementary_slackness[r_i, s_i] = abs(tt_inner_prod(X_tt, Z_tt))
            primal_res = tt_rank_reduce(tt_sub(tt_fast_matrix_vec_mul(L_op_tt, tt_reshape(X_tt, (4,))), bias_tt), eps=1e-12)
            feasibility_errors[r_i, s_i] = tt_inner_prod(primal_res, primal_res)
            dual_res = tt_rank_reduce(tt_sub(tt_fast_matrix_vec_mul(tt_transpose(L_op_tt), tt_reshape(Y_tt, (4, )), eps=1e-12), tt_rank_reduce(tt_add(tt_reshape(Z_tt, (4,)), obj_tt), eps=1e-12)), eps=1e-12)
            if T_tt is not None:
                dual_res = tt_rank_reduce(tt_sub(dual_res, tt_reshape(T_tt, (4,))), eps=1e-12)
            dual_feasibility_errors[r_i, s_i] = tt_inner_prod(dual_res, dual_res)
            num_iters[r_i, s_i] = info["num_iters"]
            ranksX[r_i, s_i, :] = info["ranksX"]
            ranksY[r_i, s_i, :] = info["ranksY"]
            ranksZ[r_i, s_i, :] = info["ranksZ"]

            # Only track T if it exists
            if ranksT is not None:
                ranksT[r_i, s_i, :] = info["ranksT"]

            print(f"Convergence after {num_iters[r_i, s_i]:.0f} iterations. "
                  f"Compl Slackness: {complementary_slackness[r_i, s_i]:.4e}. "
                  f"Feasibility error: {feasibility_errors[r_i, s_i]:.4e}. "
                  f"Dual Feasibility error: {dual_feasibility_errors[r_i, s_i]:.4e}.")
            print(f"Convergence in {runtimes[r_i, s_i]:.2f}s. Memory: {memory[r_i, s_i]:.2f} MB.")

    # Pass ranksT only if it was ever used
    print_results_summary(
        config, args,
        runtimes, problem_creation_times, num_iters,
        feasibility_errors, dual_feasibility_errors, complementary_slackness,
        ranksX, ranksY, ranksZ,
        ranksT=ranksT,
        memory=memory
    )


def format_ranks_with_std(mean_array, std_array, precision=1):
    """Helper function to format rank arrays into a 'mean±std' string."""
    if mean_array is None or std_array is None:
        return "N/A"
    
    mean_array = np.asarray(mean_array)
    std_array = np.asarray(std_array)
    
    if mean_array.size == 0:
        return "[]"
        
    formatted_parts = [f"{m:.{precision}f}±{s:.{precision}f}" for m, s in zip(mean_array, std_array)]
    return f"[{', '.join(formatted_parts)}]"

def print_results_summary(config, args, runtimes, problem_creation_times,
                          num_iters, feasibility_errors, dual_feasibility_errors, complementary_slackness,
                          ranksX, ranksY, ranksZ, ranksT=None, memory=None):
    """
    Prints a formatted summary of the experimental results, including means
    and standard deviations for performance metrics and tensor ranks in a 'mean ± std' format.
    """
    print("\n" + "=" * 80)
    print(f"{'FINAL RESULTS SUMMARY':^80}")
    print("=" * 80)
    print("Values are reported as Mean ± Standard Deviation over all seeds.\n")

    for r_i, rank in enumerate(config["max_ranks"]):
        # --- Calculate Means for Metrics ---
        mean_runtime = np.mean(runtimes[r_i, :])
        mean_creation_time = np.mean(problem_creation_times[r_i, :])
        mean_iters = np.mean(num_iters[r_i, :])
        mean_feasibility = np.mean(feasibility_errors[r_i, :])
        mean_dual_feasibility = np.mean(dual_feasibility_errors[r_i, :])
        mean_slackness = np.mean(complementary_slackness[r_i, :])

        # --- Calculate Standard Deviations for Metrics ---
        std_runtime = np.std(runtimes[r_i, :])
        std_creation_time = np.std(problem_creation_times[r_i, :])
        std_iters = np.std(num_iters[r_i, :])
        std_feasibility = np.std(feasibility_errors[r_i, :])
        std_dual_feasibility = np.std(dual_feasibility_errors[r_i, :])
        std_slackness = np.std(complementary_slackness[r_i, :])

        # --- Print Table for the Current Rank ---
        print(f"--- Rank: {rank} ---")
        print(f"  {'Metric':<28} | {'Value (Mean ± Std)':>25}")
        print(f"  {'-' * 28} | {'-' * 25}")
        print(f"  {'Solution Time (s)':<28} | {f'{mean_runtime:.3f} ± {std_runtime:.3f}':>25}")
        print(f"  {'Problem Creation (s)':<28} | {f'{mean_creation_time:.3f} ± {std_creation_time:.3f}':>25}")
        print(f"  {'Iterations':<28} | {f'{mean_iters:.1f} ± {std_iters:.1f}':>25}")
        print(f"  {'Feasibility Error':<28} | {f'{mean_feasibility:.2e} ± {std_feasibility:.2e}':>25}")
        print(f"  {'Dual Feasibility Error':<28} | {f'{mean_dual_feasibility:.2e} ± {std_dual_feasibility:.2e}':>25}")
        print(f"  {'Complementary Slackness':<28} | {f'{mean_slackness:.2e} ± {std_slackness:.2e}':>25}")

        if args.track_mem and memory is not None:
            mean_mem = np.mean(memory[r_i, :])
            std_mem = np.std(memory[r_i, :])
            print(f"  {'Peak Memory (MB)':<28} | {f'{mean_mem:.3f} ± {std_mem:.3f}':>25}")

        # --- Calculate and Print Rank Statistics ---
        print(f"  {'-' * 28} | {'-' * 25}")
        print(f"  {'Rank Statistics':<55}")

        # Ranks for X
        avg_ranks_X = np.mean(ranksX[r_i, :, :], axis=0)
        std_ranks_X = np.std(ranksX[r_i, :, :], axis=0)
        print(f"  {'  Ranks X':<26}: {format_ranks_with_std(avg_ranks_X, std_ranks_X)}")

        # Ranks for Y
        avg_ranks_Y = np.mean(ranksY[r_i, :, :], axis=0)
        std_ranks_Y = np.std(ranksY[r_i, :, :], axis=0)
        print(f"  {'  Ranks Y':<26}: {format_ranks_with_std(avg_ranks_Y, std_ranks_Y)}")

        # Ranks for Z
        avg_ranks_Z = np.mean(ranksZ[r_i, :, :], axis=0)
        std_ranks_Z = np.std(ranksZ[r_i, :, :], axis=0)
        print(f"  {'  Ranks Z':<26}: {format_ranks_with_std(avg_ranks_Z, std_ranks_Z)}")

        # Ranks for T (if provided)
        if ranksT is not None:
            avg_ranks_T = np.mean(ranksT[r_i, :, :], axis=0)
            std_ranks_T = np.std(ranksT[r_i, :, :], axis=0)
            print(f"  {'  Ranks T':<26}: {format_ranks_with_std(avg_ranks_T, std_ranks_T)}")

        print("")  # Add a newline for spacing between rank blocks

    print("=" * 80)
