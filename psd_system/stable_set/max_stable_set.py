import copy
import sys
import os
import time

sys.path.append(os.getcwd() + '/../../')

from dataclasses import dataclass
from src.tt_ops import *
from psd_system.graph_plotting import *
from src.tt_ipm import tt_ipm, _tt_get_block


@dataclass
class Config:
    seed = 3
    ranks = [3]


if __name__ == "__main__":
    print("Creating Problem...")

    np.random.seed(Config.seed)
    graph = tt_random_graph(Config.ranks)
    G = tt_scale(0.5, tt_add(graph, tt_one(len(Config.ranks) + 1, shape=(2, 2))))
    G = tt_rank_reduce(G)
    print(np.round(tt_matrix_to_matrix(G), decimals=2))

    As_tt = tt_mask_to_linear_op(G)
    bias_tt = tt_zeros(len(G), shape=(2, 2))
    J_tt = tt_one(len(G), shape=(2, 2))
    trace_constraint_tt = tt_identity(len(G))

    print("...Problem created!")
    print(f"Objective Ranks: {tt_ranks(J_tt)}")
    print(f"Constraint Ranks: As {tt_ranks(As_tt)}, bias {tt_ranks(bias_tt)}")
    t0 = time.time()
    XZ_tt, Y_tt = tt_ipm(J_tt, As_tt, bias_tt, max_iter=50)
    t1 = time.time()
    X_tt = tt_rank_reduce(_tt_get_block(0, 0, XZ_tt))
    Z_tt = tt_rank_reduce(_tt_get_block(1, 1, XZ_tt))
    print("Solution: ")
    print(np.round(tt_matrix_to_matrix(XZ_tt), decimals=2))
    print(f"Problem solved in {t1 - t0:.3f}s")
    print(f"Objective value: {tt_inner_prod(J_tt, X_tt)}")
    print("Complementary Slackness: ", tt_inner_prod(X_tt, Z_tt))
    print(f"Ranks- XZ_tt {tt_ranks(XZ_tt)} X_tt {tt_ranks(X_tt)} Z_tt {tt_ranks(Z_tt)} ")
