import copy
import sys
import os

import numpy as np

sys.path.append(os.getcwd() + '/../')
from src.tt_ops import *

def _core_op_right_from_matrix(core):
    op_core = np.zeros((core.shape[0], 4, *core.shape[1:]))
    op_core[:, 0, 0, 0] = core[:, 0, 0]
    op_core[:, 0, 0, 1] = core[:, 1, 0]
    op_core[:, 1, 0, 0] = core[:, 0, 1]
    op_core[:, 1, 0, 1] = core[:, 1, 1]
    op_core[:, 2, 1, 0] = core[:, 0, 0]
    op_core[:, 2, 1, 1] = core[:, 1, 0]
    op_core[:, 3, 1, 0] = core[:, 0, 1]
    op_core[:, 3, 1, 1] = core[:, 1, 1]
    return op_core


def tt_op_right_from_tt_matrix(matrix_tt):
    """
    Converts a matrix_tt for matrix multiplication into a linear_op_tt
    """
    return [_core_op_right_from_matrix(c) for c in matrix_tt]


def _core_op_left_from_matrix(core):
    op_core = np.zeros((core.shape[0], 4, *core.shape[1:]))
    op_core[:, 0, 0, 0] = core[:, 0, 0]
    op_core[:, 0, 1, 0] = core[:, 1, 0]
    op_core[:, 2, 0, 0] = core[:, 0, 1]
    op_core[:, 2, 1, 0] = core[:, 1, 1]
    op_core[:, 1, 0, 1] = core[:, 0, 0]
    op_core[:, 1, 1, 1] = core[:, 1, 0]
    op_core[:, 3, 0, 1] = core[:, 0, 1]
    op_core[:, 3, 1, 1] = core[:, 1, 1]
    return op_core


def tt_op_left_from_tt_matrix(matrix_tt):
    """
    Converts a matrix_tt for matrix multiplication into a linear_op_tt
    """
    return [_core_op_left_from_matrix(c) for c in matrix_tt]


def _core_op_sym_from_matrix(core):
    op_core = np.zeros((core.shape[0], 4, *core.shape[1:]))
    op_core[:, 0, 0, 0] = core[:, 0, 0]
    op_core[:, 0, 0, 1] = core[:, 1, 0]
    op_core[:, 0, 1, 0] = core[:, 1, 0]

    op_core[:, 1, 0, 0] = core[:, 0, 1]
    op_core[:, 1, 0, 1] = core[:, 0, 0] + core[:, 1, 1]
    op_core[:, 1, 1, 1] = core[:, 1, 0]

    op_core[:, 2, 1, 0] = core[:, 0, 0] + core[:, 1, 1]
    op_core[:, 2, 1, 1] = core[:, 1, 0]
    op_core[:, 2, 0, 0] = core[:, 0, 1]

    op_core[:, 3, 1, 0] = core[:, 0, 1]
    op_core[:, 3, 0, 1] = core[:, 0, 1]
    op_core[:, 3, 1, 1] = core[:, 1, 1]


    return op_core


def tt_op_sym_from_tt_matrix(matrix_tt):
    """
    Converts a matrix_tt for matrix multiplication into a linear_op_tt
    """
    return [_core_op_sym_from_matrix(c) for c in matrix_tt]


s_matrix_tt_1 = tt_random_gaussian([4], shape=(2, 2))
s_matrix_tt_1 = tt_rank_reduce(tt_add(s_matrix_tt_1, tt_transpose(s_matrix_tt_1)), err_bound=0)


s_matrix_tt_2 = tt_random_gaussian([4], shape=(2, 2))
#s_matrix_tt_2 = tt_rank_reduce(tt_add(s_matrix_tt_1, tt_transpose(s_matrix_tt_1)), err_bound=0)

print("Right")
result = tt_mat_mat_mul(s_matrix_tt_1, s_matrix_tt_2)

print(np.round(tt_matrix_to_matrix(result), decimals=2))

print("Left")
result = tt_mat_mat_mul(s_matrix_tt_2, s_matrix_tt_1)

print(np.round(tt_matrix_to_matrix(result), decimals=2))

print("Right op")

s_matrix_tt_1_op = tt_op_right_from_tt_matrix(s_matrix_tt_1)

result_1 = tt_mat(tt_linear_op(s_matrix_tt_1_op, s_matrix_tt_2), shape=(2, 2))

print(np.round(tt_matrix_to_matrix(result_1), decimals=2))

print("Left op")

s_matrix_tt_1_op = tt_op_left_from_tt_matrix(s_matrix_tt_1)

result_2 = tt_mat(tt_linear_op(s_matrix_tt_1_op, s_matrix_tt_2), shape=(2, 2))

print(np.round(tt_matrix_to_matrix(result_2), decimals=2))
print("-----")
print(np.round(tt_matrix_to_matrix(result_1) + tt_matrix_to_matrix(result_2), decimals=2))

s_matrix_tt_sym_op = tt_add(tt_op_left_from_tt_matrix(s_matrix_tt_1), tt_op_right_from_tt_matrix(s_matrix_tt_1))

result = tt_mat(tt_linear_op(s_matrix_tt_sym_op, s_matrix_tt_2), shape=(2, 2))

print(np.round(tt_matrix_to_matrix(result), decimals=2))




