import numpy as np
from typing import List, Tuple
from itertools import product
from copy import deepcopy
from cy_src.ops_cy import *



def break_core_bond(core, err_bound=1e-18):
    """ Breaks up a bond between two cores """
    shape = core.shape
    k = len(shape) // 2
    A = core.reshape(np.prod(shape[:k]), -1)
    U, S, V_T = np.linalg.svd(A)
    non_sing_eig_idxs = np.asarray(np.abs(S) > err_bound).nonzero()[0]
    if len(non_sing_eig_idxs) == 0:
        non_sing_eig_idxs = np.array([0])
    S = S[non_sing_eig_idxs]
    next_rank = len(S)
    U = U[:, non_sing_eig_idxs]
    V_T = V_T[non_sing_eig_idxs, :]
    G_i = U.reshape(*shape[:k], next_rank)
    G_ip1 = (np.diag(S) @ V_T).reshape(next_rank, *shape[k:])
    return [G_i, G_ip1]


def core_bond(core_1, core_2):
    return np.einsum("abc, cde -> abde", core_1, core_2)


def robust_cholesky(A, epsilon=1e-10):
    A = np.array(A)
    A_reg = A + epsilon * np.eye(A.shape[0])
    L = np.linalg.cholesky(A_reg)
    return L


def safe_multi_dot(matrices: List[np.array]):
    if len(matrices) > 1:
        return np.linalg.multi_dot(matrices)
    elif len(matrices) == 1:
        return matrices[0]
    return []
