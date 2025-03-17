import copy
import sys
import os
import yaml
import argparse
import tracemalloc


sys.path.append(os.getcwd() + '/../../')

from src.tt_ops import *
from src.tt_ipm import tt_ipm
import time


def tt_G_entrywise_mask_op(G):
    vec_g_copy = tt_vec(copy.deepcopy(G))
    basis = []
    for g_core in vec_g_copy:
        core = np.zeros((g_core.shape[0], 2, 2, g_core.shape[-1]))
        core[:, 0, 0] = g_core[:, 0]
        core[:, 1, 1] = g_core[:, 1]
        basis.append(core)
    return tt_rank_reduce(basis)

def tt_G_entrywise_mask_op_adj(G):
    return tt_G_entrywise_mask_op(G)


def tt_tr_op(dim):
    op =[]
    for i, c in enumerate(tt_vec(tt_identity(dim))):
        core = np.zeros((c.shape[0], 2, 2, c.shape[-1]))
        core[:, 0] = c
        op.append(core)
    return tt_rank_reduce(op)

if __name__ == "__main__":
    np.set_printoptions(linewidth=np.inf, threshold=np.inf, precision=4, suppress=True)
    parser = argparse.ArgumentParser(description="Script with optional memory tracking.")
    parser.add_argument("--track_mem", action="store_true", help="Enable memory tracking from a certain point.")
    parser.add_argument("--config", type=str, required=True, help="Path to the YAML configuration file")
    args = parser.parse_args()
    with open(os.getcwd() + '/../../' + args.config, "r") as file:
        config = yaml.safe_load(file)

    print("Creating Problem...")

    np.random.seed(config["seed"])
    G = tt_random_graph(config["dim"], config["max_rank"])
    #print(np.round(tt_matrix_to_matrix(G), decimals=2))

    # I
    G_entry_tt_op = tt_G_entrywise_mask_op(G)
    G_entry_tt_op_adjoint = tt_G_entrywise_mask_op_adj(G)

    # II
    tr_tt_op = tt_tr_op(config["dim"])
    tr_bias_tt = [E(0, 0) for _ in range(config["dim"])]

    # Objective
    J_tt = tt_one_matrix(config["dim"])

    # Constraint
    L_tt = tt_rank_reduce(tt_add(G_entry_tt_op, tr_tt_op))
    bias_tt = tr_bias_tt

    lag_maps = {"y": tt_rank_reduce(tt_diag(tt_vec(tt_sub(tt_one_matrix(config["dim"]), tt_add(G, tr_bias_tt)))))}

    print("...Problem created!")
    print(f"Objective Ranks: {tt_ranks(J_tt)}")
    print(f"Constraint Ranks: \n \t L {tt_ranks(L_tt)}, bias {tt_ranks(bias_tt)}")
    if args.track_mem:
        print("Memory tracking started...")
        tracemalloc.start()  # Start memory tracking
    t0 = time.time()
    X_tt, Y_tt, T_tt, Z_tt = tt_ipm(
        lag_maps,
        J_tt,
        L_tt,
        bias_tt,
        max_iter=config["max_iter"],
        verbose=True,
        feasibility_tol=config["feasibility_tol"],
        centrality_tol=config["centrality_tol"],
        op_tol=config["op_tol"],
        tau=config["tau"],
        direction=config["direction"]
    )
    t1 = time.time()
    if args.track_mem:
        current, peak = tracemalloc.get_traced_memory()
        print(f"Current memory usage: {current / 10 ** 6:.2f} MB")
        print(f"Peak memory usage: {peak / 10 ** 6:.2f} MB")
        tracemalloc.stop()  # Stop tracking after measuring
    #print("Solution: ")
    #print(np.round(tt_matrix_to_matrix(X_tt), decimals=4))
    print(f"Problem solved in {t1 - t0:.3f}s")
    print(f"Objective value: {tt_inner_prod(J_tt, X_tt)}")
    print("Complementary Slackness: ", tt_inner_prod(X_tt, Z_tt))
    primal_res = tt_rank_reduce(tt_sub(tt_fast_matrix_vec_mul(L_tt, tt_vec(X_tt)), tt_vec(bias_tt)), rank_weighted_error=True, eps=1e-12)
    print(f"Total primal feasibility error: {np.sqrt(tt_inner_prod(primal_res,  primal_res))}")
    print(f"Ranks X_tt {tt_ranks(X_tt)} Y_tt {tt_ranks(Y_tt)} Z_tt {tt_ranks(Z_tt)} ")
