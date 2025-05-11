import copy
import sys
import os

import numpy as np

sys.path.append(os.getcwd() + '/../')
from src.tt_ops import *
from src.tt_ops import tt_rank_reduce, tt_rl_orthogonalise
from src.tt_ipm import _ineq_step_size
from src.tt_eigen import *

np.random.seed(3)
np.set_printoptions(linewidth=np.inf, threshold=np.inf, precision=2, suppress=True)
s_matrix_tt = tt_scale(5, tt_random_gaussian([4, 4, 4], shape=(2, 2)))
s_matrix_tt = tt_rank_reduce(tt_add(s_matrix_tt, tt_transpose(s_matrix_tt)), eps=1e-8)
s_matrix_tt = tt_add(tt_fast_hadammard(s_matrix_tt, s_matrix_tt), tt_scale(0.1, tt_one_matrix(4)))
mask = [E(0, 0)] + tt_one_matrix(3)
s_matrix_tt = tt_fast_hadammard(s_matrix_tt, mask)

#print("Real min eig: ", np.min(np.linalg.eigvalsh(tt_matrix_to_matrix(copy.deepcopy(s_matrix_tt)))))

#eig_vec, eig_val = tt_min_eig(copy.deepcopy(s_matrix_tt), verbose=True)
#print("Min Eig val: ", eig_val)

s_matrix_tt_2 = tt_scale(1, tt_random_gaussian([4, 4, 4], shape=(2, 2)))
s_matrix_tt_2 = tt_rank_reduce(tt_add(s_matrix_tt_2, tt_transpose(s_matrix_tt_2)), eps=1e-8)
s_matrix_tt_2 = tt_fast_hadammard(s_matrix_tt_2, mask)
step_size, error = _ineq_step_size(s_matrix_tt, s_matrix_tt_2, 1e-8, tt_sub(tt_one_matrix(len(mask)), mask))
A = tt_matrix_to_matrix(s_matrix_tt) + tt_matrix_to_matrix(tt_sub(tt_one_matrix(len(mask)), mask))
B = tt_matrix_to_matrix(s_matrix_tt_2) + tt_matrix_to_matrix(tt_sub(tt_one_matrix(len(mask)), mask))
true_step_size = -A / B
true_step_size = min(max(np.min((-A/B)[true_step_size > 0]), 0), 1)
print("Ineq step_size: ", step_size, error, true_step_size)
print(np.min(tt_matrix_to_matrix(s_matrix_tt) + true_step_size*tt_matrix_to_matrix(s_matrix_tt_2)))
print(np.min(tt_matrix_to_matrix(s_matrix_tt) + step_size*tt_matrix_to_matrix(s_matrix_tt_2) + tt_matrix_to_matrix(tt_sub(tt_one_matrix(len(mask)), mask))))