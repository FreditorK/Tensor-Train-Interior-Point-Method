import sys
import os

sys.path.append(os.getcwd() + '/../')
from src.tt_ops import *
from src.tt_ops import tt_rank_reduce, _tt_generalised_nystroem
from src.tt_ineq_check import tt_is_geq

np.random.seed(4)

s_matrix_tt = tt_random_gaussian([4, 4, 4], shape=(2, 2))
s_matrix_tt = tt_rank_reduce(tt_add(s_matrix_tt, tt_transpose(s_matrix_tt)), eps=0)

X_tt = tt_random_gaussian([2], shape=(2, 2))
b_tt = tt_vec(tt_scale(-0.11, [np.ones((1, 2, 2, 1)) for _ in range(2)]))

check, val, res = tt_is_geq(s_matrix_tt, X_tt, b_tt)
print(check, val, res)
print(np.min(np.round(tt_vec_to_vec(tt_sub(b_tt, tt_fast_matrix_vec_mul(s_matrix_tt, tt_vec(X_tt)))), decimals=5)))

