import numpy as np
from typing import List
from itertools import product

PHI_MATRIX = np.array([[1, 1],
                       [1, -1]], dtype=float).reshape(1, 2, 2, 1)

MINUS_ONE = lambda n, rank: [-np.ones((1, 2, rank)) / 2] \
                            + [np.concatenate((np.eye(rank, rank), np.eye(rank, rank)), axis=0).reshape(rank, 2,
                                                                                                        rank)] * (n - 2) \
                            + [np.ones((rank, 2, 1)) / 2]


def tt_rl_orthogonalize(tt_train: List[np.array]):
    for idx in reversed(range(len(tt_train) - 1)):
        shape = tt_train[idx].shape
        Q_T, R = np.linalg.qr(tt_train[idx + 1].reshape(2, -1))
        tt_train[idx] = (tt_train[idx].reshape(-1, 2) @ R.T).reshape(*shape)
    return tt_train


def tt_round(tt_train: List[np.array]):
    tt_train = tt_rl_orthogonalize(tt_train)
    for idx in range(len(tt_train) - 1):
        shape = tt_train[idx].shape
        Q_T, R = np.linalg.qr(tt_train[idx].reshape(-1, 2))
        U, S, V_T = np.linalg.svd(R)
        tt_train[idx] = (tt_train[idx].reshape(-1, 2) @ U).reshape(*shape)
        shape_2 = tt_train[idx + 1].shape
        tt_train[idx + 1] = (S @ V_T @ tt_train[idx + 1].reshape(2, -1)).reshape(shape_2)
    return tt_train


def tt_svd(fourier_tensor: np.array) -> List[np.array]:
    """
    Converts a tensor into a tensor train
    """
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


def tt_to_tensor(tt_train: List[np.array]) -> np.array:
    """
    Converts a tensor train back into a tensor
    """
    dim = len(tt_train)
    fourier_tensor = np.zeros([2] * dim)
    multi_idxs = product([0, 1], repeat=dim)
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
    column_1 = np.concatenate((tensor_1, np.zeros_like(tensor_1)), axis=0)
    column_2 = np.concatenate((np.zeros_like(tensor_2), tensor_2), axis=0)
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
