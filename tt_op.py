import itertools

import numpy as np
from typing import List
from itertools import product

PHI_MATRIX = np.array([[1, 1],
                       [1, -1]], dtype=float).reshape(1, 2, 2, 1)

ONE = lambda n: [np.ones((1, 2, 1)) for _ in range(n)]

LEADING_ONE = lambda n: [np.array([1, 0], dtype=float).reshape(1, 2, 1) for _ in range(n)]


def tt_rl_orthogonalize(tt_train: List[np.array]):
    for idx in reversed(range(1, len(tt_train))):
        shape_p1 = tt_train[idx].shape
        shape = tt_train[idx - 1].shape
        Q_T, R = np.linalg.qr(tt_train[idx].reshape(shape_p1[0], -1).T)
        tt_train[idx] = Q_T.T.reshape(-1, shape_p1[1], shape_p1[-1])
        tt_train[idx - 1] = (tt_train[idx - 1].reshape(-1, R.shape[-1]) @ R.T).reshape(-1, shape[1],
                                                                                       tt_train[idx].shape[0])
    return tt_train


def tt_round(tt_train: List[np.array]):
    tt_train = tt_rl_orthogonalize(tt_train)
    rank = 1
    for idx in range(len(tt_train) - 1):
        idx_shape = tt_train[idx].shape
        next_idx_shape = tt_train[idx + 1].shape
        U, S, V_T = np.linalg.svd(tt_train[idx].reshape(rank * idx_shape[1], -1))
        non_sing_eig_idxs = np.nonzero(S)[0]
        S = S[non_sing_eig_idxs]
        next_rank = len(S)
        U = U[:, non_sing_eig_idxs]
        V_T = V_T[non_sing_eig_idxs, :]
        tt_train[idx] = U.reshape(rank, idx_shape[1], next_rank)
        tt_train[idx + 1] = (np.diag(S) @ V_T @ tt_train[idx + 1].reshape(V_T.shape[-1], -1)).reshape(next_rank,
                                                                                                      next_idx_shape[1],
                                                                                                      -1)
        rank = next_rank
    return tt_train


def tt_svd(fourier_tensor: np.array) -> List[np.array]:
    """
    Converts a tensor into a tensor train
    """
    shape = fourier_tensor.shape
    rank = 1
    cores = []
    for i in range(len(shape) - 1):
        A = fourier_tensor.reshape(rank * shape[i], -1)
        U, S, V_T = np.linalg.svd(A)
        non_sing_eig_idxs = np.nonzero(S)[0]
        S = S[non_sing_eig_idxs]
        next_rank = len(S)
        U = U[:, non_sing_eig_idxs]
        V_T = V_T[non_sing_eig_idxs, :]
        G_i = U.reshape(rank, shape[i], next_rank)
        cores.append(G_i)
        fourier_tensor = np.diag(S) @ V_T
        rank = next_rank
    G_n = fourier_tensor.reshape(rank, 2, 1)
    cores.append(G_n)
    return cores


def tt_to_tensor(tt_train: List[np.array]) -> np.array:
    """
    Converts a tensor train back into a tensor
    """
    fourier_tensor = np.zeros([t.shape[1] for t in tt_train])
    multi_idxs = product(*[list(range(t.shape[1])) for t in tt_train])
    for idx in multi_idxs:
        dot_prod = tt_train[0][:, idx[0], :]
        for i, core in enumerate(tt_train[1:]):
            dot_prod = dot_prod @ core[:, idx[i + 1], :]
        fourier_tensor[idx] = dot_prod
    return fourier_tensor


def _block_diag_tensor(tensor_1: np.array, tensor_2: np.array) -> np.array:
    """
    For internal use: Concatenates two tensors to a block diagonal tensor
    """
    column_1 = np.concatenate((tensor_1, np.zeros((tensor_2.shape[0], tensor_1.shape[1], tensor_1.shape[2]))), axis=0)
    column_2 = np.concatenate((np.zeros((tensor_1.shape[0], tensor_2.shape[1], tensor_2.shape[2])), tensor_2), axis=0)
    return np.concatenate((column_1, column_2), axis=-1)


def tt_add(tt_train_1: List[np.array], tt_train_2: List[np.array]) -> List[np.array]:
    """
    Adds two tensor trains
    """
    new_cores = [np.concatenate((tt_train_1[0], tt_train_2[0]), axis=-1)]
    for core_1, core_2 in zip(tt_train_1[1:-1], tt_train_2[1:-1]):
        H_i = _block_diag_tensor(core_1, core_2)
        new_cores.append(H_i)
    new_cores.append(np.concatenate((tt_train_1[-1], tt_train_2[-1]), axis=0))
    return new_cores


def _tt_train_kron(core_1: np.array, core_2: np.array) -> np.array:
    """
    For internal use: Computes the kronecker product between two TT-cores with appropriate dimensional
    expansion
    """
    layers = []
    for i in range(core_1.shape[1]):
        layers.append(np.kron(core_1[:, None, i, :], core_2[:, None, i, :]))
    return np.concatenate(layers, axis=1)


def tt_hadamard(tt_train_1: List[np.array], tt_train_2: List[np.array]) -> List[np.array]:
    """
    Computes the hadamard product/pointwise multiplication of two tensor trains
    """
    new_cores = []
    for core_1, core_2 in zip(tt_train_1, tt_train_2):
        new_cores.append(_tt_train_kron(core_1, core_2))
    return new_cores


def bool_to_tt_train(bool_values: List[bool]):
    """
    Converts a list of boolean values into its respective tensor train
    """
    return [np.array([1, 2 * float(b_value) - 1]).reshape(1, -1, 1) for b_value in bool_values]


def tt_inner_prod(tt_train_1: List[np.array], tt_train_2: List[np.array]) -> float:
    """
    Computes the inner product between two tensor trains
    """
    result = np.kron(tt_train_1[0][:, 0, :], tt_train_2[0][:, 0, :]) + np.kron(tt_train_1[0][:, 1, :],
                                                                               tt_train_2[0][:, 1, :])
    for core_1, core_2 in zip(tt_train_1[1:], tt_train_2[1:]):
        result = result @ (np.kron(core_1[:, 0, :], core_2[:, 0, :]) + np.kron(core_1[:, 1, :], core_2[:, 1, :]))
    return result


def tt_bool_op(tt_train: List[np.array]) -> List[np.array]:
    """
    Produces the truth table result tensor
    """
    new_cores = []
    for core in tt_train:
        new_core = np.kron(core[:, None, 0, :], PHI_MATRIX[:, 0, :, :]) + np.kron(core[:, None, 1, :],
                                                                                  PHI_MATRIX[:, 1, :, :])
        new_cores.append(new_core)
    return new_cores


def tt_xnor(tt_train_1: List[np.array], tt_train_2: List[np.array]) -> List[np.array]:
    """
    Produces the truth table result tensor
    """
    new_cores = []
    tt_train_2_permuted = [
        np.kron(np.array([[0], [1]]), core[:, None, 0, :]) + np.kron(np.array([[1], [0]]), core[:, None, 1, :]) for core
        in tt_train_2]
    tt_train_2 = [np.concatenate((np.expand_dims(t, 1), np.expand_dims(t_permuted, 1)), axis=1) for t, t_permuted in
                  zip(tt_train_2, tt_train_2_permuted)]
    for idx, (core_1, core_2) in enumerate(zip(tt_train_1, tt_train_2)):
        new_core = np.kron(core_1[:, None, 0, :], core_2[:, :, 0, :]) + np.kron(core_1[:, None, 1, :],
                                                                                core_2[:, :, 1, :])
        new_cores.append(new_core)
    return new_cores


def tt_xor(tt_train_1: List[np.array], tt_train_2: List[np.array]) -> List[np.array]:
    xor_cores = tt_xnor(tt_train_1, tt_train_2)
    xor_cores[0] *= -1
    return xor_cores


def tt_and(tt_train_1: List[np.array], tt_train_2: List[np.array]) -> List[np.array]:
    xnor_cores = tt_xnor(tt_train_1, tt_train_2)
    xnor_cores[0] *= 0.5
    tt_train_1[0] *= 0.5
    tt_train_2[0] *= 0.5
    leading_one = LEADING_ONE(len(tt_train_1))
    leading_one[0] *= -0.5
    sum_cores = tt_add(leading_one, tt_add(tt_add(xnor_cores, tt_train_1), tt_train_2))
    rounded_sum = tt_round(sum_cores)
    return rounded_sum


def tt_or(tt_train_1: List[np.array], tt_train_2: List[np.array]) -> List[np.array]:
    xnor_cores = tt_xnor(tt_train_1, tt_train_2)
    xnor_cores[0] *= -0.5
    tt_train_1[0] *= 0.5
    tt_train_2[0] *= 0.5
    leading_one = LEADING_ONE(len(tt_train_1))
    leading_one[0] *= 0.5
    sum_cores = tt_add(leading_one, tt_add(tt_add(xnor_cores, tt_train_1), tt_train_2))
    rounded_sum = tt_round(sum_cores)
    return rounded_sum
