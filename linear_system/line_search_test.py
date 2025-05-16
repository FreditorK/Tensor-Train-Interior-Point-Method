import sys
import os

import numpy as np
import scipy



sys.path.append(os.getcwd() + '/../')
from src.tt_ops import *
from src.tt_ops import tt_rank_reduce
from src.tt_eigen import tt_max_generalised_eigen

for seed in range(10,  20):
    np.random.seed(seed) # 9
    print("Seed: ", seed)
    X_tt = tt_random_gaussian([2, 9, 3, 7, 6], shape=(2, 2))
    X_tt = tt_fast_mat_mat_mul(tt_transpose(X_tt), X_tt)
    X_tt = tt_add(X_tt, tt_scale(0.01, tt_identity(len(X_tt))))

    s_matrix_tt = tt_random_gaussian([4, 2, 2, 3, 5], shape=(2, 2))
    s_matrix_tt = tt_add(tt_rank_reduce(tt_add(s_matrix_tt, tt_transpose(s_matrix_tt)), eps=0), X_tt)


    print("----")
    A = tt_matrix_to_matrix(X_tt)
    B = tt_matrix_to_matrix(s_matrix_tt)

    #print(np.linalg.eigvals(tt_matrix_to_matrix(X_tt)))
    #print(np.linalg.eigvals(tt_matrix_to_matrix(s_matrix_tt)))
    L_inv = np.linalg.inv(scip.linalg.cholesky(A, check_finite=False, lower=True))
    eig_val = np.linalg.eigvalsh(-L_inv @ B @ L_inv.T)[-1]
    step_size = 1 / eig_val
    print(step_size, np.linalg.eigvalsh(-L_inv @ B @ L_inv.T)[-2:])

    #print(tt_min_eig(tt_add(X_tt, s_matrix_tt))[0], np.linalg.eigvals(A + B))
    alpha, a = tt_max_generalised_eigen(X_tt, s_matrix_tt, verbose=True)
    print(alpha, a)
    if 0 <= step_size <= 1:
        assert abs(step_size - alpha) < 1e-5, f"UNequal {seed}!!!"
