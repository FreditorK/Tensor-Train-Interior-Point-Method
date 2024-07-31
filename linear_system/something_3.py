import sys
import os
import time
from copy import copy

import numpy as np

sys.path.append(os.getcwd() + '/../')
from dataclasses import dataclass
from src.tt_ops import *
from src.tt_ops import *


@dataclass
class Config:
    num_columns = 2
    tt_length = 6
    tt_max_rank = 5


np.random.seed(42)

import numpy as np
from src.tt_ops import _tt_op_op_collapse, _als_grad_22_sq, _als_grad_33_sq, _als_grad_44_sq

v_core = np.random.randn(2, 2, 2, 3)
V_00 = v_core[:, 0, 0, :]
C = np.random.randn(4, 9)
m, n = C.shape
p, q = V_00.shape
A = np.random.randn(n*q**2, m*p**2)
func = lambda v: np.trace(A @ np.kron(C, np.kron(v, v)))
grad_func_22 = grad(func)
func = lambda v: np.trace(A @ np.kron(np.kron(v, v), C))
grad_func_33 = grad(func)
func = lambda v: np.trace(A @ np.kron(np.kron(v, v), np.kron(v, v)))
grad_func_44 = grad(func)
print("22-Ground truth: \n", grad_func_22(V_00))
print("Mine: \n", _als_grad_22_sq(A, C, V_00))

print("33-Ground truth: \n", grad_func_33(V_00))
print("Mine: \n", _als_grad_33_sq(A, V_00, C))

print("44-Ground truth: \n", grad_func_44(V_00))
print("Mine: \n", _als_grad_44_sq(A, V_00))