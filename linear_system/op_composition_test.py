import sys
import os
import time
import numpy as np
import copy

sys.path.append(os.getcwd() + '/../')
from dataclasses import dataclass
from src.tt_ops import *


matrix = tt_random_gaussian([2, 2], shape=(2, 2))
op_1 = [np.random.randn(1, 4, 2, 2, 1) for _ in range(3)]
op_2 = [np.random.randn(1, 4, 2, 2, 1) for _ in range(3)]

op_compose = tt_op_op_compose(op_1, op_2)

result_1 = tt_linear_op(op_2, tt_mat(tt_linear_op(op_1, matrix), shape=(2, 2)))

result_2 = tt_linear_op(op_compose, matrix)

res = tt_sub(result_1, result_2)

print(tt_inner_prod(res, res))