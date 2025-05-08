import copy
import sys
import os

import numpy as np
import scipy


sys.path.append(os.getcwd() + '/../')
from src.tt_ops import *
from src.tt_ops import tt_rank_reduce, _tt_generalised_nystroem, tt_random_gaussian
from src.tt_eigen import tt_max_generalised_eigen
from src.tt_eig import tt_min_eig, tt_max_eig

np.random.seed(4) # 9
np.set_printoptions(linewidth=np.inf, threshold=np.inf, precision=4, suppress=True)
s_matrix_tt = tt_random_gaussian([4, 4], shape=(2, 2))
s_matrix_tt = tt_rank_reduce(tt_add(s_matrix_tt, tt_transpose(s_matrix_tt)), eps=0)

#X_tt = tt_random_gaussian([2, 3], shape=(2, 2))
#X_tt = tt_fast_mat_mat_mul(tt_transpose(X_tt), X_tt)
X_tt = s_matrix_tt

print("----")

sym_gauss_1 = tt_random_gaussian([2, 2], (2, 2))
sym_gauss_1 = tt_add(tt_transpose(sym_gauss_1), sym_gauss_1)
sym_gauss_2 = tt_random_gaussian([2, 2], (2, 2))
sym_gauss_2 = tt_add(tt_transpose(sym_gauss_2), sym_gauss_2)


print(tt_matrix_to_matrix(X_tt))
print(np.linalg.eigvals(tt_matrix_to_matrix(X_tt)))
X_tt_rounded = _tt_generalised_nystroem(copy.deepcopy(X_tt), tt_gaussian_1=sym_gauss_1,tt_gaussian_2=sym_gauss_2)
print(tt_matrix_to_matrix(X_tt_rounded))
print(tt_ranks(X_tt_rounded))
print(np.linalg.eigvals(tt_matrix_to_matrix(X_tt)))
diff = tt_sub(X_tt_rounded, X_tt)
print(tt_inner_prod(diff, diff))
