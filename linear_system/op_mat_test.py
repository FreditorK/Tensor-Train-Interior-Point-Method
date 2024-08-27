import sys
import os
import time
from copy import copy

import numpy as np

sys.path.append(os.getcwd() + '/../')
from dataclasses import dataclass
from src.tt_ops import *
from src.tt_ops import *

A = tt_random_gaussian([3], shape=(4, 2, 2))
x = tt_random_gaussian([2], shape=(2, 2))
Ax = tt_linear_op(A, x)

A = tt_op_to_mat(A)
x = tt_vec(x)
Ax_2 = tt_matrix_vec_mul(A, x)

res = tt_sub(Ax, Ax_2)
print(tt_inner_prod(res, res))
