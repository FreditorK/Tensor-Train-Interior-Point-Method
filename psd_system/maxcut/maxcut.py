import sys
import os
import time
import argparse
import yaml
import numpy as np

sys.path.append(os.getcwd() + '/../../')

from src.tt_ops import *
from src.tt_ipm import tt_ipm
from memory_profiler import memory_usage
from src.utils import print_results_summary


def tt_diag_constraint_op(dim):
    identity = tt_identity(dim)
    basis = tt_diag_op(identity)
    return basis, identity

def tt_obj_matrix(rank, dim):
    graph_tt = tt_rank_reduce(tt_random_graph(dim, rank))
    laplacian_tt = tt_sub(tt_diag(tt_fast_matrix_vec_mul(graph_tt, [np.ones((1, 2, 1)) for _ in range(dim)],  1e-12)), graph_tt)
    return laplacian_tt

def create_problem(dim, rank):
    print(f"Creating Problem for dim={dim}, rank={rank}...")
    scale = max(2**(dim-7), 1)
    obj_tt = tt_obj_matrix(rank, dim)
    L_tt, bias_tt = tt_diag_constraint_op(dim)
    lag_y = tt_diag_op(tt_sub(tt_one_matrix(dim), tt_identity(dim)))
    return tt_reshape(tt_normalise(obj_tt, radius=scale), (4,)), L_tt, tt_reshape(tt_normalise(bias_tt, radius=scale), (4,)), lag_y


if __name__ == "__main__":
    np.set_printoptions(linewidth=np.inf, threshold=np.inf, precision=4, suppress=True)
    parser = argparse.ArgumentParser(description="Script with optional memory tracking.")
    parser.add_argument("--track_mem", action="store_true", help="Enable memory tracking from a certain point.")
    parser.add_argument("--config", type=str, required=True, help="Path to the YAML configuration file")
    args = parser.parse_args()
    with open(os.getcwd() + '/../../' + args.config, "r") as file:
        config = yaml.safe_load(file)

    # --- Data Collection Arrays ---
    num_ranks = len(config["max_ranks"])
    num_seeds = len(config["seeds"])
    dim = config["dim"]

    problem_creation_times = np.zeros((num_ranks, num_seeds))
    runtimes = np.zeros((num_ranks, num_seeds))
    memory = np.zeros((num_ranks, num_seeds))
    complementary_slackness = np.zeros((num_ranks, num_seeds))
    feasibility_errors = np.zeros((num_ranks, num_seeds))
    num_iters = np.zeros((num_ranks, num_seeds))
    ranksX = np.zeros((num_ranks, num_seeds, dim - 1))
    ranksY = np.zeros((num_ranks, num_seeds, dim - 1))
    ranksZ = np.zeros((num_ranks, num_seeds, dim - 1))

    # --- Main Experiment Loop ---
    for r_i, rank in enumerate(config["max_ranks"]):
        print(f"\n===== Processing Rank: {rank} =====")
        for s_i, seed in enumerate(config["seeds"]):
            print(f"  --- Running Seed: {seed} ---")
            np.random.seed(seed)
            t1 = time.time()
            # In the original code, `config["max_rank"]` was used. It should likely be the current `rank`.
            obj_tt, L_tt, bias_tt, lag_y = create_problem(dim, rank)
            lag_maps = {"y": lag_y}
            t2 = time.time()
            
            # Define the IPM execution logic as a callable
            def run_ipm():
                return tt_ipm(
                    lag_maps,
                    obj_tt,
                    L_tt,
                    bias_tt,
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
                # The wrapper function for memory_usage needs to return the results
                def wrapper():
                    return run_ipm()
                
                res = memory_usage(proc=wrapper, max_usage=True, retval=True, include_children=True)
                X_tt, Y_tt, T_tt, Z_tt, info = res[1]
                memory[r_i, s_i] = res[0] - start_mem
            else:
                X_tt, Y_tt, T_tt, Z_tt, info = run_ipm()

            t3 = time.time()
            
            # --- Store Results ---
            problem_creation_times[r_i, s_i] = t2 - t1
            runtimes[r_i, s_i] = t3 - t2
            complementary_slackness[r_i, s_i] = abs(tt_inner_prod(X_tt, Z_tt))
            primal_res = tt_rank_reduce(tt_sub(tt_fast_matrix_vec_mul(L_tt, tt_reshape(X_tt, (4,))), bias_tt), eps=1e-12)
            feasibility_errors[r_i, s_i] = tt_inner_prod(primal_res, primal_res)
            num_iters[r_i, s_i] = info["num_iters"]
            ranksX[r_i, s_i, :] = info["ranksX"]
            ranksY[r_i, s_i, :] = info["ranksY"]
            ranksZ[r_i, s_i, :] = info["ranksZ"]

            # --- Per-Seed Print (Optional but good for debugging) ---
            print(f"Convergence after {num_iters[r_i, s_i]:.0f} iterations. Compl Slackness: {complementary_slackness[r_i, s_i]:.4e}. Feasibility error: {feasibility_errors[r_i, s_i]:.4e}")
            print(f"Convergence in {runtimes[r_i, s_i]}s. Memory: {memory[r_i, s_i]} MB.")

    print_results_summary(config, args, runtimes, problem_creation_times, num_iters, feasibility_errors, complementary_slackness, ranksX, ranksY, ranksZ, memory)