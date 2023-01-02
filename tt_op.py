from typing import List

import numpy as np
import scipy as sc
from itertools import product

PHI_MATRIX = np.array([[1, 1],
                       [1, -1]], dtype=float).reshape(1, 2, 2, 1)


def tt_svd(fourier_tensor: np.array):
    shape = fourier_tensor.shape
    ranks = [1] + [2] * (len(shape) - 1)
    cores = []
    for i in range(len(ranks) - 1):
        A = fourier_tensor.reshape(ranks[i] * shape[i], -1)
        U, S, V_T = np.linalg.svd(A)
        num_non_sing_eig = len(S)
        U = U[:, :num_non_sing_eig]
        V_T = V_T[:num_non_sing_eig, :]
        G_i = U.reshape(ranks[i], shape[i], ranks[i + 1])
        cores.append(G_i)
        fourier_tensor = np.diag(S) @ V_T
    G_n = fourier_tensor.reshape(2, 2, 1)
    cores.append(G_n)
    return cores


def tt_to_tensor(tt_train):
    dim = len(tt_train)
    fourier_tensor = np.zeros([2] * dim)
    multi_idxs = product([0, 1], repeat=dim)
    for idx in multi_idxs:
        dot_prod = tt_train[0][:, idx[0], :]
        for i, core in enumerate(tt_train[1:]):
            dot_prod = dot_prod @ core[:, idx[i + 1], :]
        fourier_tensor[idx] = dot_prod
    return fourier_tensor


def _block_diag_tensor(tensor_1, tensor_2):
    column_1 = np.concatenate((tensor_1, np.zeros_like(tensor_1)), axis=0)
    column_2 = np.concatenate((np.zeros_like(tensor_2), tensor_2), axis=0)
    return np.concatenate((column_1, column_2), axis=-1)


def tt_add(tt_train_1, tt_train_2):
    new_cores = [np.concatenate((tt_train_1[0], tt_train_2[0]), axis=-1)]
    for core_1, core_2 in zip(tt_train_1[1:-1], tt_train_2[1:-1]):
        H_i = _block_diag_tensor(core_1, core_2)
        new_cores.append(H_i)
    new_cores.append(np.concatenate((tt_train_1[-1], tt_train_2[-1]), axis=0))
    return new_cores


def _tt_train_kron(core_1, core_2):
    layers = []
    for i in range(core_1.shape[1]):
        layers.append(np.kron(core_1[:, None, i, :], core_2[:, None, i, :]))
    return np.concatenate(layers, axis=1)


def tt_hadamard(tt_train_1, tt_train_2):
    new_cores = []
    for core_1, core_2 in zip(tt_train_1, tt_train_2):
        new_cores.append(_tt_train_kron(core_1, core_2))
    return new_cores


def bool_to_tt_train(bool_values: List[bool]):
    return [np.array([1, 2 * float(b_value) - 1]).reshape(1, -1, 1) for b_value in bool_values]


def tt_inner_prod(tt_train_1, tt_train_2):
    result = np.kron(tt_train_1[0][:, 0, :], tt_train_2[0][:, 0, :]) + np.kron(tt_train_1[0][:, 1, :],
                                                                               tt_train_2[0][:, 1, :])
    for core_1, core_2 in zip(tt_train_1[1:], tt_train_2[1:]):
        result = result @ (np.kron(core_1[:, 0, :], core_2[:, 0, :]) + np.kron(core_1[:, 1, :], core_2[:, 1, :]))
    return result


def tt_bool_op(tt_train):
    new_cores = []
    for core in tt_train:
        new_core = np.kron(core[:, None, 0, :], PHI_MATRIX[:, 0, :, :]) + np.kron(core[:, None, 1, :], PHI_MATRIX[:, 1, :, :])
        new_cores.append(new_core)
    return new_cores
