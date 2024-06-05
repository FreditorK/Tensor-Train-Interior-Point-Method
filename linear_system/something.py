import sys
import os
import time
from copy import copy

import numpy as np

sys.path.append(os.getcwd() + '/../')
from dataclasses import dataclass
from src.tt_ops import *
from sklearn.utils.extmath import randomized_range_finder
from src.tt_ops import _tt_op_op_collapse

@dataclass
class Config:
    num_columns = 2
    tt_length = 6
    tt_max_rank = 5

np.random.seed(389)

linear_op = tt_random_gaussian_linear_op([5, 3])
a = tt_random_gaussian_linear_op([5, 3])
linear_op = tt_add(linear_op, a)
linear_op = sum([break_core_bond(c) for c in linear_op], [])
linear_op = tt_rl_orthogonalise(linear_op)
print(tt_inner_prod(linear_op, linear_op))
print(np.sqrt(np.trace(linear_op[0].reshape(linear_op[0].shape[0]*linear_op[0].shape[1], linear_op[0].shape[-1]).T @ linear_op[0].reshape(linear_op[0].shape[0]*linear_op[0].shape[1], linear_op[0].shape[-1]))))

# TODO: This below is equal to taking the inner product between two tensor matrices
"""
print(tt_inner_prod(z_sym, y_sym))
k = tt_linear_op(z_sym, y)
print(tt_inner_prod(y, k))
"""
