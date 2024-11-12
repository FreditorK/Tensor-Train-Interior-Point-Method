import sys
import os

sys.path.append(os.getcwd() + '/../')
from src.tt_ops import *
from src.tt_ops import tt_rank_reduce
from src.tt_amen import tt_amen, tt_pinv
import numpy as np
import copy

np.random.seed(4)
np.set_printoptions(linewidth=600, threshold=np.inf, precision=4, suppress=True)

s_matrix_tt = [np.array([[1.0, 0, 0, 0],
                         [0, 0, 0, 0],
                         [0, 0, 0, 0],
                         [0, 0, 1.0, 0.0]]).reshape(1, 4, 4, 1) for _ in range(2)]#tt_random_gaussian([2], shape=(4, 4))

print(np.round(tt_matrix_to_matrix(s_matrix_tt), 3))

x = tt_scale(0.9, tt_random_gaussian([2], shape=(2, 2)))

b = tt_mat(tt_linear_op([s.reshape(s.shape[0], 4, 2, 2, s.shape[-1]) for s in s_matrix_tt], x), shape=(2, 2))

sol, res = tt_pinv(copy.copy(s_matrix_tt), 1e-8)
print(tt_ranks(sol))
pseudo_inv = sol #tt_rank_reduce(tt_mat_mat_mul(sol, tt_transpose(s_matrix_tt)), err_bound=1e-8)
print([c.shape for c in pseudo_inv])
print([c.shape for c in b])

#pseudo_inv = s_matrix_tt
ra = tt_mat(tt_linear_op([s.reshape(s.shape[0], 4, 2, 2, s.shape[-1]) for s in pseudo_inv], b), shape=(2, 2))
ra = tt_sub(ra, x)
print("Err", tt_inner_prod(ra, ra))
print(np.round(tt_matrix_to_matrix(pseudo_inv), 3))

res = tt_sub(s_matrix_tt, tt_mat_mat_mul(s_matrix_tt, tt_mat_mat_mul(pseudo_inv, s_matrix_tt)))
print(tt_inner_prod(res, res))
#sol = [c.reshape(c.shape[0], 4, 2, 2, c.shape[-1]) for c in sol]
#print(np.round(tt_matrix_to_matrix(tt_mat(tt_linear_op(op, tt_mat(tt_linear_op(sol, tt_mat(b_tt, shape=(2, 2))), shape=(2, 2))), shape=(2, 2))), decimals=2))
#print(np.round(tt_matrix_to_matrix(tt_mat(b_tt, shape=(2, 2))), decimals=2))