import copy
import sys
import os

import yaml
import argparse

sys.path.append(os.getcwd() + '/../../')

from src.tt_ipm import *
from memory_profiler import memory_usage
import time
from src.utils import print_results_summary


def tt_G_entrywise_mask_op(G):
    vec_g_copy = tt_split_bonds(copy.deepcopy(G))
    basis = []
    for g_core in vec_g_copy:
        core = np.zeros((g_core.shape[0], 2, 2, g_core.shape[-1]))
        core[:, 0, 0] = g_core[:, 0]
        core[:, 1, 1] = g_core[:, 1]
        basis.append(core)
    return tt_rank_reduce(tt_reshape(basis, (4, 4)))

def tt_tr_constraint(dim):
    op =[]
    for i, c in enumerate(tt_split_bonds([np.eye(2).reshape(1, 2, 2, 1) for _ in range(dim)])):
        core = np.zeros((c.shape[0], 2, 2, c.shape[-1]))
        core[:, 0] = c
        op.append(core)
    return tt_rank_reduce(tt_reshape(op, (4, 4))), [E(0, 0) for _ in range(config["dim"])]

def tt_obj_matrix(dim):
    return tt_one_matrix(dim)


def create_problem(dim, rank):
    print("Creating Problem...")
    scale = max(2**(dim-6), 1)
    G = tt_rank_reduce(tt_random_graph(dim, rank))
    obj_tt = tt_obj_matrix(dim)
    L_tt, bias_tt = tt_tr_constraint(dim)
    L_tt = tt_rank_reduce(tt_add(L_tt, tt_G_entrywise_mask_op(G)))
    lag_y = tt_rank_reduce(tt_diag_op(tt_sub(tt_one_matrix(config["dim"]), tt_add(G, bias_tt))))
    return tt_reshape(tt_normalise(obj_tt, radius=scale), (4,)), L_tt, tt_reshape(tt_normalise(bias_tt, radius=8*scale), (4,)), lag_y

if __name__ == "__main__":
    np.set_printoptions(linewidth=np.inf, threshold=np.inf, precision=4, suppress=True)
    parser = argparse.ArgumentParser(description="Script with optional memory tracking.")
    parser.add_argument("--track_mem", action="store_true", help="Enable memory tracking from a certain point.")
    parser.add_argument("--config", type=str, required=True, help="Path to the YAML configuration file")
    args = parser.parse_args()
    with open(os.getcwd() + '/../../' + args.config, "r") as file:
        config = yaml.safe_load(file)

    # --- Data Collection Arrays (using the improved NumPy structure) ---
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

    # --- Main Experiment Loop (with nested loops for ranks and seeds) ---
    for r_i, rank in enumerate(config["max_ranks"]):
        print(f"\n===== Processing Rank: {rank} =====")
        for s_i, seed in enumerate(config["seeds"]):
            print(f"  --- Running Seed: {seed} ---")
            np.random.seed(seed)
            t1 = time.time()
            # Use the 'rank' from the loop, not from config
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
                def wrapper():
                    return run_ipm()
                
                res = memory_usage(proc=wrapper, max_usage=True, retval=True, include_children=True)
                X_tt, Y_tt, T_tt, Z_tt, info = res[1]
                memory[r_i, s_i] = res[0] - start_mem
            else:
                X_tt, Y_tt, T_tt, Z_tt, info = run_ipm()

            t3 = time.time()
            
            # --- Store Results in NumPy arrays using (r_i, s_i) indices ---
            problem_creation_times[r_i, s_i] = t2 - t1
            runtimes[r_i, s_i] = t3 - t2
            complementary_slackness[r_i, s_i] = abs(tt_inner_prod(X_tt, Z_tt))
            primal_res = tt_rank_reduce(tt_sub(tt_fast_matrix_vec_mul(L_tt, tt_reshape(X_tt, (4, ))), bias_tt), eps=1e-12)
            feasibility_errors[r_i, s_i] = tt_inner_prod(primal_res,  primal_res)
            num_iters[r_i, s_i] = info["num_iters"]
            # Saving the rank arrays, which was missing previously
            ranksX[r_i, s_i, :] = info["ranksX"]
            ranksY[r_i, s_i, :] = info["ranksY"]
            ranksZ[r_i, s_i, :] = info["ranksZ"]

            print(f"Convergence after {num_iters[r_i, s_i]:.0f} iterations. Compl Slackness: {complementary_slackness[r_i, s_i]:.4e}. Feasibility error: {feasibility_errors[r_i, s_i]:.4e}")
            print(f"Convergence in {runtimes[r_i, s_i]}s. Memory: {memory[r_i, s_i]} MB.")

    print_results_summary(config, args, runtimes, problem_creation_times, num_iters, feasibility_errors, complementary_slackness, ranksX, ranksY, ranksZ, memory)