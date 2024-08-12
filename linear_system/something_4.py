import sys
import os
import time
from copy import copy

import numpy as np

sys.path.append(os.getcwd() + '/../')
from src.tt_ops import *
np.random.seed(29)

A_1 = tt_random_gaussian([2], shape=(2, 2))
A_2 = tt_random_gaussian([2], shape=(2, 2))
A_1A_2_comp = tt_linear_op_compose(A_1, A_2)
A_1A_2_comp = [c.reshape(c.shape[0], 4, *c.shape[3:]) for c in A_1A_2_comp]
A_1_op = tt_compose_to_op(A_1)
#print([c.shape for c in A_1_op])
#A_1A_2_comp_by_op = tt_eval_constraints(A_1_op, A_2)

#diff = tt_sub(A_1A_2_comp, A_1A_2_comp_by_op)
#print(tt_inner_prod(diff, diff))
A_1_op = [c.reshape(c.shape[0], 2, 2, *c.shape[-3:]) for c in A_1_op]
print([c.shape for c in A_1_op])
np.set_printoptions(threshold=np.inf, linewidth=800)
print(np.round(tt_op_to_matrix(A_1), decimals=2))
printable_matrix = np.zeros((16, 16))
for (i1, j1, i2, j2) in product([0, 1], [0, 1], [0, 1], [0, 1]):
    sub_A = [A_1_op[0][:, i1, j1], A_1_op[1][:, i2, j2]]
    column_index = i2 * 1 + i1 * 2
    row_index = j2 * 1 + j1 * 2
    printable_matrix[row_index*4: row_index*4 + 4, column_index*4: column_index*4 + 4] = np.round(tt_op_to_matrix(sub_A), decimals=2)
print(printable_matrix)

c_0 = np.array([[1, 0], [1, 0]]).reshape(1, 2, 2, 1)
c_1 = np.array([[0, 1], [0, 1]]).reshape(1, 2, 2, 1)
"""
printable_matrix = np.zeros((16, 16))
choice = [c_0, c_1]
for (i1, j1, i2, j2) in product([0, 1], [0, 1], [0, 1], [0, 1]):
    sub_A = [choice[i1], choice[i2]]
    column_index = i2 * 1 + i1 * 2
    row_index = j2 * 1 + j1 * 2
    printable_matrix[row_index*4: row_index*4 + 4, column_index*4: column_index*4 + 4] = np.round(tt_op_to_matrix(sub_A), decimals=2)
print(printable_matrix)
"""
"""
concat_c = np.concatenate(
    (
        np.expand_dims(c_0, axis=1),
        np.expand_dims(c_0, axis=1),
        np.expand_dims(c_1, axis=1),
        np.expand_dims(c_1, axis=1)
    ), axis=1
)
special_tt = [concat_c for _ in range(2)]
A_1_op = [c.reshape(c.shape[0], 2, 2, *c.shape[-3:]) for c in special_tt]
printable_matrix = np.zeros((16, 16))
for (i1, j1, i2, j2) in product([0, 1], [0, 1], [0, 1], [0, 1]):
    sub_A = [A_1_op[0][:, i1, j1], A_1_op[1][:, i2, j2]]
    column_index = i2 * 1 + i1 * 2
    row_index = j2 * 1 + j1 * 2
    printable_matrix[row_index*4: row_index*4 + 4, column_index*4: column_index*4 + 4] = np.round(tt_op_to_matrix(sub_A), decimals=2)
print(printable_matrix)
"""