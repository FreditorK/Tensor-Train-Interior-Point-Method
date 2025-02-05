import copy
import sys
import os

sys.path.append(os.getcwd() + '/../../')

from dataclasses import dataclass
from src.tt_ops import *
from src.tt_ipm import tt_ipm, _tt_get_block
import time
from src.tt_eig import tt_min_eig, tt_max_eig
from max_stable_set import *
from src.regular_ipm import ipm


if __name__ == "__main__":
    np.set_printoptions(linewidth=np.inf, threshold=np.inf, precision=4, suppress=True)

    print("Creating Problem...")

    np.random.seed(Config.seed)
    G = tt_random_graph(Config.dim, Config.max_rank)
    #print(np.round(tt_matrix_to_matrix(G), decimals=2))

    # I
    G_entry_tt_op = tt_G_entrywise_mask_op(G)
    G_entry_tt_op_adjoint = tt_G_entrywise_mask_op_adj(G)

    # II
    tr_tt_op = tt_tr_op(Config.dim)
    tr_tt_op_adjoint = tt_tr_op_adjoint(Config.dim)
    tr_bias_tt = [np.array([[1.0, 0.0], [0.0, 0.0]]).reshape(1, 2, 2, 1) for _ in range(Config.dim)]

    # Objective
    J_tt = tt_one_matrix(Config.dim)

    # Constraint
    L_tt = tt_rank_reduce(tt_add(G_entry_tt_op, tr_tt_op))
    L_tt_adjoint = tt_rank_reduce(tt_add(G_entry_tt_op_adjoint, tr_tt_op_adjoint))
    bias_tt = tr_bias_tt

    lag_maps = {"y": tt_rank_reduce(tt_diag(tt_vec(tt_sub(tt_one_matrix(Config.dim), tt_add(G, tr_bias_tt)))))}

    print("...Problem created!")
    print(f"Objective Ranks: {tt_ranks(J_tt)}")
    print(f"Constraint Ranks: \n \t L {tt_ranks(L_tt)}, L^* {tt_ranks(L_tt_adjoint)}, bias {tt_ranks(bias_tt)}")
    t0 = time.time()
    X, Y, _, Z = ipm(
        lag_maps,
        J_tt,
        L_tt,
        bias_tt,
        max_iter=16,
        verbose=True
    )
    t1 = time.time()
    print("Solution: ")
    print(np.round(X, decimals=2))
    print(f"Problem solved in {t1 - t0:.3f}s")
    print(f"Objective value: {np.trace(tt_matrix_to_matrix(J_tt).T @ X)}")
    print("Complementary Slackness: ", np.trace(X.T @ Z))