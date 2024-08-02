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

v_core = np.random.randn(4, 2, 2, 3)
V_00 = v_core[:, 0, 0]
V_01 = v_core[:, 0, 1]
V_10 = v_core[:, 1, 0]
V_11 = v_core[:, 1, 1]

truth = _tt_op_op_collapse(v_core, np.swapaxes(np.copy(v_core), axis1=1, axis2=2))
truth_00 = truth[:, 0, 0]
truth_01 = truth[:, 0, 1]
truth_10 = truth[:, 1, 0]
truth_11 = truth[:, 1, 1]

newV_00 = np.kron(V_00, V_00) + np.kron(V_10, V_10)
newV_01 = np.kron(V_01, V_00) + np.kron(V_11, V_10)
newV_10 = np.kron(V_00, V_01) + np.kron(V_10, V_11)
newV_11 = np.kron(V_01, V_01) + np.kron(V_11, V_11)

print(np.sum(np.abs(truth_00 - newV_00)))
print(np.sum(np.abs(truth_01 - newV_01)))
print(np.sum(np.abs(truth_10 - newV_10)))
print(np.sum(np.abs(truth_11 - newV_11)))


"""
V_00 = v_core[:, 0, 0, :]
C = np.random.randn(2, 2)
m, n = C.shape
p, q = V_00.shape
A = np.random.randn(n*q**2, m*p**2)
func = lambda v: np.trace(A @ np.kron(C, np.kron(v, v)))
grad_func_22 = grad(func)
func = lambda v: np.trace(A @ np.kron(np.kron(v, v), C))
grad_func_33 = grad(func)
func = lambda v: np.trace(A @ np.kron(np.kron(v, v), np.kron(v, v)))
grad_func_44 = grad(func)
print(A.shape, C.shape, V_00.shape)
print("22-Ground truth: \n", grad_func_22(V_00))
print("Mine: \n", _als_grad_22_sq(A, C, V_00))

print("33-Ground truth: \n", grad_func_33(V_00))
print("Mine: \n", _als_grad_33_sq(A, V_00, C))

print("44-Ground truth: \n", grad_func_44(V_00))
print("Mine: \n", _als_grad_44_sq(A, V_00))
"""