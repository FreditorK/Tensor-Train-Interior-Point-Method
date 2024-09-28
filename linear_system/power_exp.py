import sys
import os

import numpy as np
from torch.nn.functional import threshold

sys.path.append(os.getcwd() + '/../')
from src.tt_ops import *
from src.tt_ops import _tt_core_collapse, _tt_lr_random_orthogonalise, _tt_matrix_vec_mul, \
    tt_randomised_min_eigentensor, tt_rank_reduce

np.random.seed(7)

linear_op = tt_random_gaussian([4], shape=(2, 2))
linear_op = tt_rank_reduce(tt_add(linear_op, tt_transpose(linear_op)))

psd_matrix = tt_random_gaussian([4], shape=(2, 2))
psd_matrix = tt_rank_reduce(tt_add(linear_op, tt_transpose(linear_op))) #tt_mat_mat_mul(psd_matrix, tt_transpose(psd_matrix))
psd_matrix = tt_rank_reduce(psd_matrix, err_bound=0)


def tt_psd_exp(matrix_tt: List[np.array], num_iter=1500, conv_tol=1e-8, error_tol=1e-8):
    """
    Only for symmetric matrices
    """
    target_ranks = [1] + tt_ranks(matrix_tt) + [1]  #[int(np.ceil(np.sqrt(r))) + 1 for r in tt_ranks(matrix_tt)] + [1]
    gaussian_tt = [
        np.divide(1, l_n * 2 * l_np1) * np.random.randn(l_n, 2, l_np1)
        for i, (l_n, l_np1) in enumerate(zip(target_ranks[:-1], target_ranks[1:]))
    ]
    n = len(matrix_tt)
    normalisation = np.sqrt(tt_inner_prod(matrix_tt, matrix_tt))
    matrix_tt = tt_scale(-np.divide(1, normalisation), matrix_tt)
    identity = tt_identity(n)
    identity = tt_scale(2, identity)
    matrix_tt = tt_rank_reduce(tt_add(identity, matrix_tt), err_bound=error_tol)
    eig_vec_tt = tt_normalise([np.random.randn(1, 2, 1) for _ in range(n)])
    sq_root = np.inf
    mask_1 = np.array([1, 0]).reshape(1, 2, 1)
    mask_2 = np.array([0, 1]).reshape(1, 2, 1)
    for i in range(num_iter):
        prev_sq_root = sq_root
        eig_vec_tt = _tt_lr_random_orthogonalise(_tt_matrix_vec_mul(matrix_tt, eig_vec_tt), gaussian_tt)
        norm_vec = safe_multi_dot([_tt_core_collapse(c, c) for c in eig_vec_tt[1:]])
        e_1 = mask_1 * eig_vec_tt[0]
        e_2 = mask_2 * eig_vec_tt[0]
        norm_1 = _tt_core_collapse(e_1, e_1) @ norm_vec
        norm_2 = _tt_core_collapse(e_2, e_2) @ norm_vec
        sq_root = np.sqrt(norm_1 + norm_2)
        eig_vec_tt[0] *= np.divide(1, np.sqrt(np.array([norm_1, norm_2]))).reshape(1, 2, 1)
        if np.less_equal(np.abs(sq_root - prev_sq_root), conv_tol):
            print(norm_1, norm_2)
            return normalisation*(2-np.sqrt(norm_1)), normalisation*(2-np.sqrt(norm_2))
    return normalisation * (2 - norm_1), normalisation * (2 - norm_2)

block_matrix = [np.array([[1, 0], [0, 0]]).reshape(1, 2, 2, 1)] + psd_matrix
block_matrix = tt_add(block_matrix, [np.array([[0, 0], [0, 1]]).reshape(1, 2, 2, 1)] + linear_op)

eig_val_1, eig_val_2 = tt_psd_exp(block_matrix)
print(eig_val_1, eig_val_2)
_, min_eig = tt_randomised_min_eigentensor(psd_matrix, 1500, tol=1e-8)
print(min_eig)
_, min_eig = tt_randomised_min_eigentensor(linear_op, 1500, tol=1e-8)
print(min_eig)
_, min_eig = tt_randomised_min_eigentensor(block_matrix, 1500, tol=1e-8)
print(min_eig)
print(np.min(np.linalg.eigvals(tt_matrix_to_matrix(block_matrix))))
#vals, vecs = np.linalg.eig(tt_matrix_to_matrix(block_matrix))
#np.set_printoptions(linewidth=np.inf, threshold=np.inf)
#print(vals)
#print(vecs)
