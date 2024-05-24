import sys
import os
import time
from copy import copy

import numpy as np

sys.path.append(os.getcwd() + '/../')
from dataclasses import dataclass
from src.tt_op import *
from sklearn.utils.extmath import randomized_range_finder
from src.tt_op import _tt_op_op_collapse

@dataclass
class Config:
    num_columns = 2
    tt_length = 6
    tt_max_rank = 5

np.random.seed(10)

a = np.array([0.49, 0.2, 0.6])

for _ in range(15):
    a = 3*a**2 - 2*a**3
print(a)

# TODO: This below is equal to taking the inner product between two tensor matrices
"""
print(tt_inner_prod(z_sym, y_sym))
k = tt_linear_op(z_sym, y)
print(tt_inner_prod(y, k))
"""
