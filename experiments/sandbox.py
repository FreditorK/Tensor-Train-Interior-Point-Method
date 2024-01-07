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

A = np.array([[1, 2, 1], [4, 1, -1]])
B = np.array([[2, 1], [-2, -1], [3, 2]])
result = A @ B
print(result, np.sum(np.diagonal(result)))
m = 0
factors = np.zeros(2)
K = 50000
for _ in range(K):
    vec_1 = np.random.randn(1, 3)
    vec_2 = np.random.randn(1, 2)
    a = (vec_2 @ A @ vec_1.T).item()
    b = (vec_1 @ B @ vec_2.T).item()
    factors += np.array([a, b])
    m += a * b
print(factors/K)
print(m/K)