import sys
import os

import scipy



sys.path.append(os.getcwd() + '/../')
from src.tt_ops import *
from src.tt_ops import tt_rank_reduce
from src.tt_ineq_check import tt_pd_line_search, psd_projection
from src.tt_eig import tt_min_eig, tt_max_eig

np.random.seed(4) # 9

s_matrix_tt = tt_random_gaussian([4, 4], shape=(2, 2))
s_matrix_tt = tt_rank_reduce(tt_add(s_matrix_tt, tt_transpose(s_matrix_tt)), eps=0)

#X_tt = tt_random_gaussian([2, 3], shape=(2, 2))
#X_tt = tt_fast_mat_mat_mul(tt_transpose(X_tt), X_tt)
X_tt = s_matrix_tt

print("----")
A = tt_matrix_to_matrix(X_tt)
B = tt_matrix_to_matrix(s_matrix_tt)

print(np.linalg.eigvals(A))

X_tt = psd_projection(X_tt)
print(np.linalg.eigvals(tt_matrix_to_matrix(X_tt)))
