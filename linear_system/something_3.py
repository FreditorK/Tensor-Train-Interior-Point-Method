import sys
import os
import time
from copy import copy

import numpy as np

sys.path.append(os.getcwd() + '/../')
from dataclasses import dataclass
from src.tt_ops import *
from src.tt_ops import *
from memory_profiler import profile, memory_usage


@dataclass
class Config:
    num_columns = 2
    tt_length = 6
    tt_max_rank = 5


np.random.seed(42)

import numpy as np
from src.tt_ops import _tt_op_op_collapse, _als_grad_22_sq, _als_grad_33_sq, _als_grad_44_sq, _als_grad_22, _als_grad_33

v_core = np.random.randn(3, 2, 2, 2)
V_00 = v_core[:, 0, 0]
V_01 = v_core[:, 0, 1]
V_10 = v_core[:, 1, 0]
V_11 = v_core[:, 1, 1]

c_core = np.random.randn(3, 2, 2, 2)
C_00 = v_core[:, 0, 0]
C_01 = v_core[:, 0, 1]
C_10 = v_core[:, 1, 0]
C_11 = v_core[:, 1, 1]

truth = _tt_op_op_collapse(v_core, np.swapaxes(np.copy(v_core), axis1=1, axis2=2))
truth_00 = truth[:, 0, 0]
truth_01 = truth[:, 0, 1]
truth_10 = truth[:, 1, 0]
truth_11 = truth[:, 1, 1]

pair_00 = np.kron(V_00, V_00) + np.kron(V_10, V_10)
pair_01 = np.kron(V_01, V_00) + np.kron(V_11, V_10)
pair_10 = np.kron(V_00, V_01) + np.kron(V_10, V_11)
pair_11 = np.kron(V_01, V_01) + np.kron(V_11, V_11)

print(np.sum(np.abs(truth_00 - pair_00)))
print(np.sum(np.abs(truth_01 - pair_01)))
print(np.sum(np.abs(truth_10 - pair_10)))
print(np.sum(np.abs(truth_11 - pair_11)))

m, n = C_00.shape
p, q = V_00.shape
p *= p
q *= q

cc_y = np.array([[i + (m + p) * j for i in range(m)] for j in range(m)]).flatten()
cc_x = np.array([[i + (n + q) * j for i in range(n)] for j in range(n)]).flatten()

cv_y = np.array([[i + (m + p) * j for i in range(p)] for j in range(m)]).flatten()
cv_x = np.array([[i + (n + q) * j for i in range(q)] for j in range(n)]).flatten()

vc_y = np.array([[i + (m + p) * j for i in range(m)] for j in range(p)]).flatten()
vc_x = np.array([[i + (n + q) * j for i in range(n)] for j in range(q)]).flatten()

vv_y = np.array([[i + (m + p) * j for i in range(p)] for j in range(p)]).flatten()
vv_x = np.array([[i + (n + q) * j for i in range(q)] for j in range(q)]).flatten()

outer_contraction = np.random.randn((n + q) ** 2, (m + p) ** 2)

A_11 = outer_contraction[np.ix_(cc_x, cc_y)]
A_22 = outer_contraction[np.ix_(cv_x + n, cv_y + m)]
A_33 = outer_contraction[np.ix_(n * (n + q) + vc_x, m * (m + p) + vc_y)]
A_44 = outer_contraction[np.ix_(vv_x + n + n * (n + q), vv_y + m + m * (m + p))]

check_1 = np.trace(A_11 @ (np.kron(C_00, C_00) + np.kron(C_01, C_01) + np.kron(C_10, C_10) + np.kron(C_11, C_11)))


def check_2(V_00, V_10, V_01, V_11):
    pair_00 = np.kron(V_00, V_00) + np.kron(V_10, V_10)
    pair_01 = np.kron(V_01, V_00) + np.kron(V_11, V_10)
    pair_10 = np.kron(V_00, V_01) + np.kron(V_10, V_11)
    pair_11 = np.kron(V_01, V_01) + np.kron(V_11, V_11)
    return np.trace(
        A_22 @ (np.kron(C_00, pair_00) + np.kron(C_01, pair_01) + np.kron(C_10, pair_10) + np.kron(C_11, pair_11)))


check_2_grad = grad(lambda v: check_2(v, V_10, V_01, V_11))


def check_3(V_00, V_10, V_01, V_11):
    pair_00 = np.kron(V_00, V_00) + np.kron(V_10, V_10)
    pair_01 = np.kron(V_01, V_00) + np.kron(V_11, V_10)
    pair_10 = np.kron(V_00, V_01) + np.kron(V_10, V_11)
    pair_11 = np.kron(V_01, V_01) + np.kron(V_11, V_11)
    return np.trace(
        A_33 @ (np.kron(pair_00, C_00) + np.kron(pair_01, C_01) + np.kron(pair_10, C_10) + np.kron(pair_11, C_11)))


check_3_grad = grad(lambda v: check_3(v, V_10, V_01, V_11))


def check_4(V_00, V_10, V_01, V_11):
    pair_00 = np.kron(V_00, V_00) + np.kron(V_10, V_10)
    pair_01 = np.kron(V_01, V_00) + np.kron(V_11, V_10)
    pair_10 = np.kron(V_00, V_01) + np.kron(V_10, V_11)
    pair_11 = np.kron(V_01, V_01) + np.kron(V_11, V_11)
    return np.trace(A_44 @ (
        np.kron(pair_00, pair_00) + np.kron(pair_01, pair_01) + np.kron(pair_10, pair_10) + np.kron(pair_11,
                                                                                                    pair_11)))


check_4_grad = grad(lambda v: check_4(v, V_10, V_01, V_11))

"""
mem_2, vec_00 = memory_usage((_als_grad_22_sq, (A_22, C_00, C_01, C_10, V_00, V_01)), retval=True, interval=0.1,
                             timeout=1)

mem_1, true_vec_00 = memory_usage((check_2_grad, (V_00,)), retval=True, interval=0.1, timeout=1)

print("---A_22---")
print(vec_00)
print(f"My Mem usage: {max(mem_2)} MiB")

print(true_vec_00)
print(f"Autograd Mem usage: {max(mem_1)} MiB")
"""

mem_2, vec_00 = memory_usage((_als_grad_33_sq, (A_33, C_00, C_01, C_10, V_00, V_01)), retval=True, interval=0.1,
                             timeout=1)

mem_1, true_vec_00 = memory_usage((check_3_grad, (V_00,)), retval=True, interval=0.1, timeout=1)

print("---A_33---")
print(vec_00)
print(f"My Mem usage: {max(mem_2)} MiB")

print(true_vec_00)
print(f"Autograd Mem usage: {max(mem_1)} MiB")

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
