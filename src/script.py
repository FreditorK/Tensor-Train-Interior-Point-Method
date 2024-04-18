import random

import numpy as np

from tt_op import *
from tt_op import _tt_core_collapse


np.random.seed(14)
n_1 = np.array([[2.3, 3], [5, -2]])
N_1 = tt_svd(n_1)
n_2 = np.array([[1.3, -1], [-0.5, -2.3]])
N_2 = tt_svd(n_2)

combined_n_1_n_2 = np.vstack((n_1.flatten(), n_2.flatten()))


def tt_gram(tt_tensor):
    last_core = np.linalg.multi_dot([_tt_core_collapse(core, core) for core in tt_tensor[1:]])
    return [tt_tensor[0], last_core.reshape(tt_tensor[0].shape[-1], -1, 1)]


combined_N1_N2 = tt_tensor_matrix([N_1, N_2])
# combined_N1_N2 = tt_randomise_orthogonalise(combined_N1_N2, [2, 4, 3, 7, 2])

Gram_matrix = tt_gram(combined_N1_N2)
b = np.array([2, 3]).reshape(2, 1)

index_vec = tt_conjugate_gradient(Gram_matrix, [b.reshape(1, 2, 1)])
print("Index Vector", index_vec)


def tt_index(tt_index, tt_tensor):
    if len(tt_index) > 1:
        first_matrix = np.linalg.multi_dot([_tt_core_collapse(core_index, core) for core_index, core in zip(tt_index, tt_tensor[:len(tt_index)])])
    else:
        first_matrix = _tt_core_collapse(tt_index[0], tt_tensor[0])
    first_core = np.einsum("ab, bcd -> acd", first_matrix, tt_tensor[len(tt_index)])
    return [first_core] + tt_tensor[len(tt_index)+1:]

X_sol = tt_index(index_vec, combined_N1_N2)
print(b)
print(tt_partial_inner_prod(combined_N1_N2, X_sol, reversed=True))

