import sys
import os
import time
import argparse
import tracemalloc
import yaml

sys.path.append(os.getcwd() + '/../../')

from src.tt_ops import *
from src.tt_ipm import tt_ipm


def tt_diag_op(dim):
    identity = tt_identity(dim)
    basis = tt_diag(tt_vec(identity))
    return basis

def tt_diag_op_adj(dim):
    return tt_diag_op(dim)


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
    G_tt = tt_rank_reduce(tt_random_graph(config["dim"], config["max_rank"]))
    L_tt = tt_diag_op(config["dim"])
    bias_tt = tt_identity(config["dim"])

    lag_maps = {"y": tt_rank_reduce(tt_diag(tt_vec(tt_sub(tt_one_matrix(config["dim"]), tt_identity(config["dim"])))))}

    print("...Problem created!")
    print(f"Objective Ranks: {tt_ranks(G_tt)}")
    print(f"Constraint Ranks: As {tt_ranks(L_tt)}, bias {tt_ranks(bias_tt)}")
    if args.track_mem:
        print("Memory tracking started...")
        tracemalloc.start()  # Start memory tracking
    t0 = time.time()
    X_tt, Y_tt, _, Z_tt = tt_ipm(
        lag_maps,
        G_tt,
        L_tt,
        bias_tt,
        max_iter=config["max_iter"],
        verbose=True,
        feasibility_tol=config["feasibility_tol"],
        centrality_tol=config["centrality_tol"],
        op_tol=config["op_tol"]
    )
    t1 = time.time()
    if args.track_mem:
        current, peak = tracemalloc.get_traced_memory()
        print(f"Current memory usage: {current / 10 ** 6:.2f} MB")
        print(f"Peak memory usage: {peak / 10 ** 6:.2f} MB")
        tracemalloc.stop()  # Stop tracking after measuring
    #print("Solution: ")
    #print(np.round(tt_matrix_to_matrix(X_tt), decimals=3))
    print(f"Problem solved in {t1 - t0:.3f}s")
    print(f"Objective value: {tt_inner_prod(G_tt, X_tt)}")
    print("Complementary Slackness: ", tt_inner_prod(X_tt, Z_tt))
    primal_res = tt_rank_reduce(tt_sub(tt_fast_matrix_vec_mul(L_tt, tt_vec(X_tt)), tt_vec(bias_tt)), eps=1e-10)
    print(f"Total primal feasibility error: {np.sqrt(np.abs(tt_inner_prod(primal_res, primal_res)))}")
    print(f"Ranks X_tt: {tt_ranks(X_tt)}, Y_tt: {tt_ranks(Y_tt)}, Z_tt: {tt_ranks(Z_tt)} ")