import copy
import sys
import os

sys.path.append(os.getcwd() + '/../../')

from dataclasses import dataclass
from src.tt_ops import *
from psd_system.graph_plotting import *
from src.tt_ipm import tt_ipm, _tt_get_block
import time


@dataclass
class Config:
    seed = 3
    ranks = [3]


if __name__ == "__main__":
    print("Creating Problem...")

    np.random.seed(Config.seed)
    graph = tt_random_graph(Config.ranks)
    G = tt_scale(0.5, tt_add(graph, tt_one_matrix(len(Config.ranks) + 1)))
    G = tt_rank_reduce(G)
    print(np.round(tt_matrix_to_matrix(G), decimals=2))

    As_tt = tt_mask_to_linear_op(G)
    tr_tt = [np.zeros((1, 4, 2, 2, 1)) for _ in range(len(G))]
    for c in tr_tt:
        c[:, 0, :, :, 0] = np.eye(2)
        c[:, 3, :, :, 0] = np.eye(2)
    tr_bias_tt = [np.eye(2).reshape(1, 2, 2, 1) for _ in range(len(G))]
    bias_tt = tt_zero_matrix(len(G))
    J_tt = tt_one_matrix(len(G))

    print("...Problem created!")
    print(f"Objective Ranks: {tt_ranks(J_tt)}")
    print(f"Constraint Ranks: \n \t As {tt_ranks(As_tt)}, bias {tt_ranks(bias_tt)} \n \t As {tt_ranks(tr_tt)}, bias {tt_ranks(tr_bias_tt)}")
    t0 = time.time()
    X_tt, Y_tt, Z_tt = tt_ipm(
        J_tt,
        As_tt,
        bias_tt,
        tr_tt,
        tr_bias_tt,
        verbose=True)
    t1 = time.time()
    print("Solution: ")
    print(f"Problem solved in {t1 - t0:.3f}s")
    print(f"Objective value: {tt_inner_prod(J_tt, X_tt)}")
    print("Complementary Slackness: ", tt_inner_prod(X_tt, Z_tt))
    print(f"Ranks- X_tt {tt_ranks(X_tt)} Y_tt {tt_ranks(Y_tt)} Z_tt {tt_ranks(Z_tt)} ")

