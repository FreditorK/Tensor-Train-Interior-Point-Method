import os
import sys

sys.path.append(os.getcwd() + '/../')

from typing import List

import numpy as np
import jax.numpy as jnp
from itertools import product
from src.tt_op import *


def _tt_core_collapse(core_1: np.array, core_2: np.array):
    return sum([
        np.kron(core_1[(slice(None),) + i], core_2[(slice(None),) + i])
        for i in product(*([[0, 1]] * (len(core_1.shape) - 2)))
    ])


def matrix_normalise(m):
    g = np.random.randn(m.shape[0], m.shape[0])

    return (1/np.linalg.norm(g))*g @ m


def tt_inner_prod_norm(tt_train_1: List[np.array], tt_train_2: List[np.array]) -> float:
    """
    Computes the inner product between two tensor trains
    """
    return jnp.sum(
        jnp.linalg.multi_dot(
            [matrix_normalise(_tt_core_collapse(core_1, core_2)) for core_1, core_2 in zip(tt_train_1, tt_train_2)])
    )


target_ranks = [1, 2, 3, 2, 2, 5, 6, 1]
noise_train = tt_rl_orthogonalize(tt_normalise(
    [(1 / (l_n * 2 * l_np1)) * np.random.randn(l_n, 2, l_np1) for l_n, l_np1 in
     zip(target_ranks[:-1], target_ranks[1:])]))
noise_train_2 = tt_add_noise(noise_train, target_ranks[1:-1])
noise_train_3 = tt_rl_orthogonalize(tt_normalise(
    [(1 / (l_n * 2 * l_np1)) * np.random.randn(l_n, 2, l_np1) for l_n, l_np1 in
     zip(target_ranks[:-1], target_ranks[1:])]))
print(tt_inner_prod(noise_train, noise_train_2), tt_inner_prod(noise_train, noise_train_3))
print(tt_inner_prod_norm(noise_train, noise_train_2), tt_inner_prod_norm(noise_train_2, noise_train_3))
