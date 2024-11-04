import sys
import os

sys.path.append(os.getcwd() + '/../')
from src.tt_ops import *
from src.tt_ops import tt_rank_reduce
from src.tt_amen import tt_amen
import numpy as np

np.random.seed(4)
np.set_printoptions(linewidth=600, threshold=np.inf, precision=4, suppress=True)

s_matrix_tt = tt_random_gaussian([4], shape=(2, 2))
s_matrix_tt = tt_rank_reduce(tt_add(s_matrix_tt, tt_transpose(s_matrix_tt)), err_bound=0)


op = tt_op_right_from_tt_matrix(s_matrix_tt)
op = tt_op_to_mat(op)
op_expand =  tt_rank_reduce([np.eye(4).reshape(1, 4, 4, 1) for _ in range(len(s_matrix_tt))] + op)

b_tt = [c.reshape(1, 4, 1) for c in tt_identity(len(op_expand))]

sol, res = tt_amen(op_expand, b_tt, kickrank=2, verbose=True)
print(res)

true_res = tt_sub(tt_matrix_vec_mul(op_expand, sol), b_tt)
print(tt_inner_prod(true_res, true_res))
print(np.round(tt_matrix_to_matrix(tt_mat(tt_matrix_vec_mul(op_expand, sol), shape=(2, 2))), decimals=2))
#print(np.round(tt_matrix_to_matrix(tt_mat(b_tt, shape=(2, 2))), decimals=2))