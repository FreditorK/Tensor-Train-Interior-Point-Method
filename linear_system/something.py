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

linear_op = tt_rl_orthogonalise(tt_random_gaussian_linear_op([2]))
linear_op = tt_add(linear_op, tt_transpose(linear_op))
print(np.round(linear_op[0], decimals=3))
print(np.round(linear_op[1], decimals=3))

# TODO: This below is equal to taking the inner product between two tensor matrices
"""
print(tt_inner_prod(z_sym, y_sym))
k = tt_linear_op(z_sym, y)
print(tt_inner_prod(y, k))
"""
