import copy
import sys
import os

import numpy as np


sys.path.append(os.getcwd() + '/../../')

from graph_matching import *
from src.regular_ipm import ipm
import time

if __name__ == "__main__":

    print("...Problem created!")
    print(f"Objective TT-ranks: {tt_ranks(C_tt)}")
    print(f"Eq Op-rank: {tt_ranks(L_op_tt)}")
    print(f"Eq Op-adjoint-rank: {tt_ranks(L_op_tt_adj)}")
    print(f"Eq Bias-rank: {tt_ranks(eq_bias_tt)}")
    print("-----------------------------------")
    print(f"Ineq Op-rank: {tt_ranks(Q_ineq_op)}")
    print(f"Ineq Op-adjoint-rank: {tt_ranks(Q_ineq_op_adj)}")
    print(f"Ineq Bias-rank: {tt_ranks(Q_ineq_bias)}")
    t0 = time.time()
    X, Y, T, Z = ipm(
        lag_maps,
        C_tt,
        L_op_tt,
        eq_bias_tt,
        Q_ineq_op,
        Q_ineq_bias,
        max_iter=18,
        verbose=True
    )
    t1 = time.time()
    print("Solution: ")
    print(np.round(X, decimals=2))
    print(f"Objective value: {np.trace(tt_matrix_to_matrix(C_tt).T @ X)}")
    print("Complementary Slackness: ", np.trace(X.T @ Z))
    print(f"Time: {t1 - t0}s")
    # TODO: Something went wrong when generalising to n=2

