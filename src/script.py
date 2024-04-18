import random

import numpy as np

from tt_op import *
from tt_op import _tt_core_collapse


def random_tt_train(length):
    length -= 2
    cores = [-3 * np.random.rand(1, 2, np.random.randint(5))]
    for _ in range(length):
        r = cores[-1].shape[-1]
        cores.append(np.random.rand(r, 2, np.random.randint(5)))
    r = cores[-1].shape[-1]
    cores.append(5 * np.random.randn(r, 2, 1) - 1)
    return cores


np.random.seed(14)
n_1 = np.array([[2.3, 3], [5, -2]])
N_1 = tt_svd(n_1)
n_2 = np.array([[1.3, -1], [-0.5, -2.3]])
N_2 = tt_svd(n_2)

combined_n_1_n_2 = np.vstack((n_1.flatten(), n_2.flatten()))


def tt_gram(tt_tensor):
    last_core = np.linalg.multi_dot([_tt_core_collapse(core, core) for core in tt_tensor[1:]])
    return [tt_tensor[0], last_core.reshape(tt_tensor[0].shape[-1], -1, 1)]


combined_N1_N2 = tt_merge(N_1, N_2)
# combined_N1_N2 = tt_randomise_orthogonalise(combined_N1_N2, [2, 4, 3, 7, 2])
# print(tt_partial_inner_prod(combined_N1_N2, X, reversed=True))

print(combined_n_1_n_2 @ combined_n_1_n_2.T)
reversed_tt_tensor = [c.reshape(c.shape[-1], -1, c.shape[0]) for c in reversed(combined_N1_N2)]
print(tt_partial_inner_prod(N_1, combined_N1_N2, reversed=True))
print(tt_partial_inner_prod(N_2, combined_N1_N2, reversed=True))
# K = tt_partial_inner_prod(combined_N1_N2[1:], combined_N1_N2, reversed=True)
# print(K)
core_a, core_b, = tt_gram(combined_N1_N2)
Gram_matrix = (core_a @ core_b).reshape(2, 2)
inv_gram_matrix = np.linalg.inv(Gram_matrix)
b = np.array([2, 3]).reshape(2, 1)
index_vec = (inv_gram_matrix @ b).reshape(1, 2, 1)


def tt_index(tt_index, tt_tensor):
    if len(tt_index) > 1:
        first_matrix = np.linalg.multi_dot([_tt_core_collapse(core_index, core) for core_index, core in zip(tt_index, tt_tensor[:len(tt_index)])])
    else:
        first_matrix = _tt_core_collapse(tt_index[0], tt_tensor[0])
    print(first_matrix.shape)
    first_core = np.einsum("ab, bcd -> acd", first_matrix, tt_tensor[len(tt_index)])
    return [first_core] + tt_tensor[len(tt_index)+1:]

print(tt_index([index_vec], [core_a, core_b]))

X_sol = tt_index([index_vec], combined_N1_N2)
print([c.shape for c in X_sol])
print([c.shape for c in N_1])
print([c.shape for c in combined_N1_N2])
print(tt_inner_prod(N_1, X_sol))
print(tt_inner_prod(N_2, X_sol))

