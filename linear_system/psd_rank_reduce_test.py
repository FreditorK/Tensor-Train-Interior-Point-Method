import sys
import os

import scipy



sys.path.append(os.getcwd() + '/../')
from src.tt_ops import *
from src.tt_ops import tt_rank_reduce
from src.tt_eigen import tt_max_generalised_eigen
import copy

for seed in range(0,  10):
    np.random.seed(seed) # 9
    print("Seed: ", seed)
    X_tt = tt_random_gaussian([2, 9, 3, 7, 6], shape=(2, 2))
    X_tt = tt_fast_mat_mat_mul(tt_transpose(X_tt), X_tt)

    X = tt_matrix_to_matrix(X_tt)
    print("Before", np.min(np.linalg.eigvalsh(X)), tt_ranks(X_tt))
    C_tt = tt_rank_reduce(copy.copy(X_tt), 1e-4)
    D_tt = tt_psd_rank_reduce(copy.copy(X_tt), 1e-4)
    print("After", np.min(np.linalg.eigvalsh(tt_matrix_to_matrix(C_tt))), tt_ranks(C_tt))
    print("Error C: ", np.linalg.norm(tt_matrix_to_matrix(C_tt) - X), tt_ranks(C_tt))
    print("Error D: ", np.linalg.norm(tt_matrix_to_matrix(D_tt) - X), tt_ranks(D_tt))
