import os
import sys

sys.path.append(os.getcwd() + '/../')

from typing import List

import numpy as np
import jax.numpy as jnp
from itertools import product
from src.tt_op import *

"""
def proj(tt_h, tt_n, func_result, bias):
    tt_n = tt_normalise(tt_n)
    alpha = jnp.sqrt((1 - bias ** 2) / (1 - func_result ** 2))
    beta = bias + alpha * func_result
    tt_h = tt_add(tt_mul_scal(alpha, tt_h), tt_mul_scal(-beta, tt_n))
    return tt_h


target_ranks = [1, 3, 3, 2, 3, 3, 5, 4, 2, 2, 1]
noise_train_1 = tt_rl_orthogonalize(tt_normalise(
    [(1 / (l_n * 2 * l_np1)) * np.random.randn(l_n, 2, l_np1) for l_n, l_np1 in
     zip(target_ranks[:-1], target_ranks[1:])]))

target_ranks = [1, 3, 3, 2, 3, 3, 5, 4, 2, 2, 1]
noise_train_2 = tt_normalise(
    [(1 / (l_n * 2 * l_np1)) * np.random.randn(l_n, 2, l_np1) for l_n, l_np1 in
     zip(target_ranks[:-1], target_ranks[1:])])

print(len(noise_train_2))
func_result = tt_inner_prod(noise_train_1, noise_train_2)
print(func_result)
noise_train_1_projection = proj(noise_train_1, noise_train_2, func_result, 0)
print(tt_inner_prod(noise_train_1_projection, noise_train_2))
print(tt_normed_inner_prod(noise_train_1_projection, noise_train_2))

t = [0]*len(noise_train_1_projection)
for _ in range(100):
    ks = tt_randomised_inner_prod_2(noise_train_1_projection, noise_train_1_projection)
    t = [a + k for a, k in zip(t, ks)]
print([a/1000 for a in t])
"""

A = np.array([[-3, 2], [4, 1]])
B = np.array([[1, 2], [4, -1]])
C = np.array([[2, 5], [4, -0.1]])
m = 0
n = 0
factors = []
K = 100000
for _ in range(K):
    vec_1 = np.random.randn(2, 1)
    vec_2 = np.random.randn(2, 1)
    vec_3 = np.random.randn(2, 1)
    a = (vec_1.T @ A @ vec_2).item()
    b = (vec_2.T @ B @ vec_3).item()
    c = (vec_3.T @ C @ vec_1).item()
    factors.append([a, b, c])
print(np.trace(A @ B @ C))
print(np.mean(np.prod(factors, axis=1)))
factors = np.array(factors).T
print(np.trace(A @ A.T), np.trace(B @ B.T), np.trace(C @ C.T))
Cov_1 = np.cov(factors)
print(Cov_1)