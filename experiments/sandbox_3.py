import os
import sys

sys.path.append(os.getcwd() + '/../')

from typing import List

import numpy as np
import jax.numpy as jnp
from itertools import product
from src.tt_op import *
from src.operators import partial_D

A = 0.001 * np.array([[-3, 2], [4, 1]])
B = 0.002 * np.array([[1, 2], [4, -1]])
C = 0.002 * np.array([[2, 5], [4, -0.1]])
m = 0
n = 0
result = np.trace(A @ B @ C)
print(result)


def factor_func(theta, A, B, C):
    A = jnp.block([[theta, jnp.zeros((1, A.shape[1]))], [jnp.zeros((A.shape[0], 1)), -A]])
    B = jnp.block([[theta, jnp.zeros((1, B.shape[1]))], [jnp.zeros((B.shape[0], 1)), B]])
    C = jnp.block([[theta, jnp.zeros((1, C.shape[1]))], [jnp.zeros((C.shape[0], 1)), C]])
    return jnp.abs(jnp.trace(A @ B @ C))


derivative = partial_D(factor_func, 0)


theta = 1.0
print(factor_func(theta, A, B, C))
for _ in range(100):
    d = derivative(theta, A, B, C)
    print(theta, d)
    theta -= 0.5*d
print("truth: ", result**(1/3))

"""
factors = []
K = 10000
for _ in range(K):
    vec_1 = 2 * np.round(np.random.rand(3, 1)) - 1
    vec_2 = 2 * np.round(np.random.rand(3, 1)) - 1
    vec_3 = 2 * np.round(np.random.rand(3, 1)) - 1
    a = (vec_1.T @ A @ vec_2).item()
    b = (vec_2.T @ B @ vec_3).item()
    c = (vec_3.T @ C @ vec_1).item()
    factors.append([a, b, c])
print(np.mean(np.prod(factors, axis=1)))
print(np.array(factors))
"""
