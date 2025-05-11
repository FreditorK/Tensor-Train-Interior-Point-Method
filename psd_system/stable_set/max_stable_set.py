import copy
import sys
import os

import yaml
import argparse

sys.path.append(os.getcwd() + '/../../')

from src.tt_ops import *
from src.tt_ipm import tt_ipm
from memory_profiler import memory_usage
import time


def tt_G_entrywise_mask_op(G):
    vec_g_copy = tt_split_bonds(copy.deepcopy(G))
    basis = []
    for g_core in vec_g_copy:
        core = np.zeros((g_core.shape[0], 2, 2, g_core.shape[-1]))
        core[:, 0, 0] = g_core[:, 0]
        core[:, 1, 1] = g_core[:, 1]
        basis.append(core)
    return tt_rank_reduce(basis)

def tt_tr_op(dim):
    op =[]
    for i, c in enumerate(tt_split_bonds(tt_identity(dim))):
        core = np.zeros((c.shape[0], 2, 2, c.shape[-1]))
        core[:, 0] = c
        op.append(core)
    return tt_rank_reduce(op)

def tt_obj_matrix(dim):
    return [0.5*np.ones((1, 2, 2, 1)) for _ in range(dim)]

if __name__ == "__main__":
    np.set_printoptions(linewidth=np.inf, threshold=np.inf, precision=4, suppress=True)
    parser = argparse.ArgumentParser(description="Script with optional memory tracking.")
    parser.add_argument("--track_mem", action="store_true", help="Enable memory tracking from a certain point.")
    parser.add_argument("--config", type=str, required=True, help="Path to the YAML configuration file")
    parser.add_argument('--rank', type=int, required=False, help='An integer input', default=0)
    args = parser.parse_args()
    with open(os.getcwd() + '/../../' + args.config, "r") as file:
        config = yaml.safe_load(file)
        problem_creation_times = []
        runtimes = []
        memory = []
        complementary_slackness = []
        feasibility_errors = []
        num_iters = []
        for seed in config["seeds"]:
            print("Seed: ", seed)
            np.random.seed(seed)
            t0 = time.time()
            rank = config["max_rank"] if args.rank == 0 else args.rank
            G = tt_rank_reduce(tt_random_graph(config["dim"], rank))
            t1 = time.time()
            G_entry_tt_op = tt_G_entrywise_mask_op(G)
            tr_tt_op = tt_tr_op(config["dim"])
            tr_bias_tt = [E(0, 0) for _ in range(config["dim"])]
            J_tt = tt_obj_matrix(config["dim"])
            lag_maps = {
                "y": tt_rank_reduce(tt_diag_op(tt_sub(tt_one_matrix(config["dim"]), tt_add(G, tr_bias_tt))))
            }
            L_tt = tt_rank_reduce(tt_add(G_entry_tt_op, tr_tt_op))
            bias_tt = tr_bias_tt

            J_tt = tt_reshape(J_tt, (4,))
            L_tt = tt_reshape(L_tt, (4, 4))
            bias_tt = tt_reshape(bias_tt, (4,))

            t2 = time.time()
            if args.track_mem:
                start_mem = memory_usage(max_usage=True)

                def wrapper():
                    X_tt, Y_tt, T_tt, Z_tt, info = tt_ipm(
                        lag_maps,
                        J_tt,
                        L_tt,
                        bias_tt,
                        max_iter=config["max_iter"],
                        verbose=config["verbose"],
                        feasibility_tol=config["feasibility_tol"],
                        centrality_tol=config["centrality_tol"],
                        op_tol=config["op_tol"],
                        aho_direction=False
                    )
                    return X_tt, Y_tt, T_tt, Z_tt, info


                res = memory_usage(proc=wrapper, max_usage=True, retval=True)
                X_tt, Y_tt, T_tt, Z_tt, info = res[1]
                memory.append(res[0] - start_mem)
            else:
                X_tt, Y_tt, T_tt, Z_tt, info = tt_ipm(
                    lag_maps,
                    J_tt,
                    L_tt,
                    bias_tt,
                    max_iter=config["max_iter"],
                    verbose=config["verbose"],
                    feasibility_tol=config["feasibility_tol"],
                    centrality_tol=config["centrality_tol"],
                    op_tol=config["op_tol"],
                    aho_direction=False
                )
            t3 = time.time()
            runtimes.append(t3 - t2)
            problem_creation_times.append(t2 - t1)
            complementary_slackness.append(abs(tt_inner_prod(X_tt, Z_tt)))
            primal_res = tt_rank_reduce(tt_sub(tt_fast_matrix_vec_mul(L_tt, tt_reshape(X_tt, (4, ))), bias_tt),
                                        rank_weighted_error=True, eps=1e-12)

            feasibility_errors.append(tt_inner_prod(primal_res,  primal_res))
            num_iters.append(info["num_iters"])
            print(f"Converged after {num_iters[-1]:.1f} iterations", flush=True)
            print(f"Problem created in {problem_creation_times[-1]:.3f}s", flush=True)
            print(f"Problem solved in {runtimes[-1]:.3f}s", flush=True)
            if args.track_mem:
                print(f"Peak memory {memory[-1]:.3f} MB", flush=True)
            print(f"Complementary Slackness: {complementary_slackness[-1]}", flush=True)
            print(f"Total feasibility error: {feasibility_errors[-1]}", flush=True)
        print("--- Run Summary ---", flush=True)
        print(f"Converged after avg {np.mean(num_iters):.1f} iterations", flush=True)
        print(f"Problem created in avg {np.mean(problem_creation_times):.3f}s", flush=True)
        print(f"Problem solved in avg {np.mean(runtimes):.3f}s", flush=True)
        if args.track_mem:
            print(f"Peak memory avg {np.mean(memory):.3f} MB", flush=True)
        print(f"Complementary Slackness avg: {np.mean(complementary_slackness)}", flush=True)
        print(f"Total feasibility error  avg: {np.mean(feasibility_errors)}", flush=True)