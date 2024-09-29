import sys
import os

import numpy as np
from torch.nn.functional import threshold

sys.path.append(os.getcwd() + '/../')
from src.tt_ops import *
from src.tt_ops import _tt_core_collapse, _tt_lr_random_orthogonalise, \
    tt_randomised_min_eigentensor, tt_rank_reduce

np.random.seed(7)

linear_op = tt_random_gaussian([4, 4], shape=(2, 2))
linear_op = tt_rank_reduce(tt_add(linear_op, tt_transpose(linear_op)))

psd_matrix = tt_random_gaussian([4, 4], shape=(2, 2))
psd_matrix = tt_rank_reduce(tt_mat_mat_mul(linear_op, tt_transpose(linear_op)), err_bound=0) #tt_mat_mat_mul(psd_matrix, tt_transpose(psd_matrix))

psd_matrix_approx_rl = [copy.copy(psd_matrix)[0]] + tt_generalised_nystroem(copy.copy(psd_matrix)[1:], [4, 4])
psd_matrix_approx_lr = tt_rank_retraction(copy.copy(psd_matrix), [4, 4])
print(tt_ranks(psd_matrix_approx_rl), tt_ranks(psd_matrix_approx_lr))
print(tt_l2_dist(psd_matrix, psd_matrix_approx_rl),tt_l2_dist(psd_matrix, psd_matrix_approx_lr))
#print(np.linalg.eigvals(tt_matrix_to_matrix(psd_matrix)))
#print(np.linalg.eigvals(tt_matrix_to_matrix(psd_matrix_approx)))


A = np.round(tt_matrix_to_matrix(psd_matrix), decimals=2)
A_approx_rl = np.round(tt_matrix_to_matrix(psd_matrix_approx_rl), decimals=2)
A_approx_lr = np.round(tt_matrix_to_matrix(psd_matrix_approx_lr), decimals=2)

print(A)
print(A_approx_rl)
print(A_approx_lr)