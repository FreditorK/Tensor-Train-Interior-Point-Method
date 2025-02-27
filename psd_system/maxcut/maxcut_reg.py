import sys
import os
import time

sys.path.append(os.getcwd() + '/../../')

from dataclasses import dataclass
from src.tt_ops import *
from src.regular_ipm import ipm
from maxcut import *


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Script with optional memory tracking.")
    parser.add_argument("--track_mem", action="store_true", help="Enable memory tracking from a certain point.")
    args = parser.parse_args()

    np.random.seed(Config.seed)
    np.set_printoptions(linewidth=np.inf, threshold=np.inf, precision=4, suppress=True)
    print("Creating Problem...")
    G_tt = tt_rank_reduce(tt_random_graph(Config.dim, Config.max_rank))
    #G_tt = tt_sub(tt_scale(2, G_tt), tt_one_matrix(Config.dim))
    #print(np.round(tt_matrix_to_matrix(G_tt), decimals=2))
    L_tt = tt_diag_op(Config.dim)
    bias_tt = tt_identity(Config.dim)

    lag_maps = {"y": tt_rank_reduce(tt_diag(tt_vec(tt_sub(tt_one_matrix(Config.dim), tt_identity(Config.dim)))))}

    print("...Problem created!")
    print(f"Objective Ranks: {tt_ranks(G_tt)}")
    print(f"Constraint Ranks: As {tt_ranks(L_tt)}, bias {tt_ranks(bias_tt)}")
    if args.track_mem:
        print("Memory tracking started...")
        tracemalloc.start()  # Start memory tracking
    t0 = time.time()
    X, Y, _, Z = ipm(lag_maps, G_tt, L_tt, bias_tt, max_iter=18, verbose=True)
    t1 = time.time()
    if args.track_mem:
        current, peak = tracemalloc.get_traced_memory()
        print(f"Current memory usage: {current / 10 ** 6:.2f} MB")
        print(f"Peak memory usage: {peak / 10 ** 6:.2f} MB")
        tracemalloc.stop()  # Stop tracking after measuring
    print("Solution: ")
    print(np.round(X, decimals=2))
    print(f"Objective value: {np.trace(tt_matrix_to_matrix(G_tt).T @ X)}")
    print("Complementary Slackness: ", np.trace(X.T @ Z))
    print(f"Time: {t1-t0}s")