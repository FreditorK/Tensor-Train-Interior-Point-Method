import sys
import os

sys.path.append(os.getcwd() + '/../')
from src.tt_ops import *
from src.tt_ops import tt_rank_reduce
from src.tt_amen import tt_amen, tt_pinv, tt_inv_precond
import numpy as np
import copy

np.random.seed(4)
np.set_printoptions(linewidth=600, threshold=np.inf, precision=4, suppress=True)

s_matrix_tt = [np.array([[1.0, 0, 0, 0],
                         [0, 0, 0, 0],
                         [0, 0, 0, 0],
                         [0, 0, 1.0, 0.0]]).reshape(1, 4, 4, 1) for _ in range(2)]
s_matrix_tt_2 = [np.array([[0, 0, 1, 0],
                         [0, 0, 0, 0],
                         [1, 0, 0, 0],
                         [0, 0, 0, 0.0]]).reshape(1, 4, 4, 1) for _ in range(2)]

s_matrix_tt = tt_random_gaussian([2], (2, 2)) #tt_rank_reduce(tt_add(s_matrix_tt, s_matrix_tt_2)) #tt_random_gaussian([2], (2, 2))

print(np.round(tt_matrix_to_matrix(s_matrix_tt), 3))
print("Cond", np.linalg.cond(tt_matrix_to_matrix(s_matrix_tt)))
print(tt_ranks(s_matrix_tt))
sol = tt_inv_precond(copy.copy(s_matrix_tt), [3], verbose=True)
print(np.round(tt_matrix_to_matrix(tt_mat_mat_mul(sol, s_matrix_tt)), decimals=2))
print("Cond", np.linalg.cond(tt_matrix_to_matrix(tt_mat_mat_mul(sol, s_matrix_tt))))
print(tt_ranks(sol))
pseudo_inv = sol
print([c.shape for c in pseudo_inv])
res = tt_sub(s_matrix_tt, tt_mat_mat_mul(s_matrix_tt, tt_mat_mat_mul(pseudo_inv, s_matrix_tt)))
print(tt_inner_prod(res, res))
res = tt_sub(pseudo_inv, tt_mat_mat_mul(pseudo_inv, tt_mat_mat_mul(s_matrix_tt, pseudo_inv)))
print(tt_inner_prod(res, res))
