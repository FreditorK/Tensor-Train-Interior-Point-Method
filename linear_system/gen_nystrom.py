import copy
import sys
import os

import numpy as np
from pandas.plotting import scatter_matrix
from torch.nn.functional import threshold
from unicodedata2 import decimal

sys.path.append(os.getcwd() + '/../')
from src.tt_ops import *
from src.tt_ops import _tt_core_collapse, _tt_lr_random_orthogonalise, \
    tt_randomised_min_eigentensor, tt_rank_reduce, _tt_generalised_nystroem, _tt_mat_core_collapse

np.random.seed(693)

psd_matrix_tt = tt_random_gaussian([4, 4], shape=(2, 2))
psd_matrix_tt = tt_rank_reduce(tt_mat_mat_mul(psd_matrix_tt, tt_transpose(psd_matrix_tt)), err_bound=0)
Delta_tt = tt_random_gaussian([4, 4], shape=(2, 2))
Delta_tt = tt_rank_reduce(tt_add(Delta_tt, tt_transpose(Delta_tt)), err_bound=0)

alpha = 0.5
psd_matrix_tt_add = tt_add(psd_matrix_tt, tt_scale(-alpha, Delta_tt))



"""
def tt_max_eigentensor_part(matrix_tt: List[np.array], num_iter=10, tol=1e-12):
    normalisation = np.sqrt(tt_inner_prod(matrix_tt, matrix_tt))
    matrix_tt[0] *= np.divide(1, normalisation)
    matrix_tt = tt_rank_reduce(matrix_tt)
    target_ranks = tt_ranks(matrix_tt)
    eig_vec_tt_part = tt_normalise([np.random.randn(1, 2, 1) for _ in range(len(matrix_tt)-1)])
    eig_vec_tt_part = [_tt_mat_core_collapse(m, c) for m, c in zip(matrix_tt[:-1], eig_vec_tt_part)]
    i = 0
    M = 0
    for i in range(num_iter):
        prev_M = M
        eig_vec_tt_part = [_tt_mat_core_collapse(m, c) for m, c in zip(matrix_tt[:-1], eig_vec_tt_part)]
        eig_vec_tt_part, M = tt_rank_retraction_part(eig_vec_tt_part, target_ranks)
        inner_prod_vec = safe_multi_dot([_tt_core_collapse(c, c) for c in eig_vec_tt_part])
        norm_2 = np.sqrt(inner_prod_vec @ inner_prod_vec.T)
        eig_vec_tt_part = tt_scale(np.divide(1, norm_2), eig_vec_tt_part)
        if np.less_equal(np.linalg.norm(M - prev_M), tol):
            break
    core = _tt_mat_core_collapse(matrix_tt[-1], np.random.randn(1, 2, 1))
    shape = core.shape
    for j in range(i):
        core_unfolded = M @ _tt_mat_core_collapse(matrix_tt[-1], core).reshape(M.shape[-1], -1)
        core = (core_unfolded / np.linalg.norm(core_unfolded)).reshape(*shape)
    eig_vec_tt = eig_vec_tt_part + [core]
    eig_vec_tt = tt_normalise(eig_vec_tt)
    prev_eig_vec = eig_vec_tt
    eig_vec_tt = tt_matrix_vec_mul(matrix_tt, eig_vec_tt)
    eig_val = tt_inner_prod(prev_eig_vec, eig_vec_tt)
    return tt_normalise(eig_vec_tt), normalisation * eig_val


def tt_min_eigentensor_part(matrix_tt: List[np.array], num_iter=10, tol=1e-12):
    n = len(matrix_tt)
    normalisation = np.sqrt(tt_inner_prod(matrix_tt, matrix_tt))
    identity = tt_identity(n)
    identity = tt_scale(2*normalisation, identity)
    matrix_tt = tt_sub(identity, matrix_tt)
    matrix_tt = tt_rl_orthogonalise(tt_rank_reduce(matrix_tt, err_bound=0))
    target_ranks = tt_ranks(matrix_tt)
    eig_vec_tt_init = tt_rl_orthogonalise(tt_normalise([np.random.randn(1, 2, 1) for _ in range(n)]))
    eig_vec_tt_part = [_tt_mat_core_collapse(m, c) for m, c in zip(matrix_tt[:-1], eig_vec_tt_init[:-1])]
    M = 0
    for i in range(num_iter):
        prev_M = M
        eig_vec_tt_part = [_tt_mat_core_collapse(m, c) for m, c in zip(matrix_tt[:-1], eig_vec_tt_part)]
        eig_vec_tt_part, M = tt_rank_retraction_part(eig_vec_tt_part, target_ranks)
        inner_prod_vec = safe_multi_dot([_tt_core_collapse(c, c) for c in eig_vec_tt_part]).T
        norm_2 = np.sqrt(inner_prod_vec.T @ inner_prod_vec)
        eig_vec_tt_part[0] *= np.divide(1, norm_2)
        if np.less_equal(np.linalg.norm(M - prev_M), tol):
            break
    core = _tt_mat_core_collapse(matrix_tt[-1], eig_vec_tt_init[-1])
    shape = core.shape
    prev_core_unfolded = 0
    for j in range(num_iter):
        core_unfolded = M @ _tt_mat_core_collapse(matrix_tt[-1], core).reshape(M.shape[-1], -1)
        if np.less_equal(np.linalg.norm(core_unfolded - prev_core_unfolded), tol):
            break
        prev_core_unfolded = core_unfolded
        core = (core_unfolded / np.linalg.norm(core_unfolded)).reshape(*shape)
    eig_vec_tt = eig_vec_tt_part + [core]
    eig_vec_tt = tt_normalise(eig_vec_tt)
    prev_eig_vec = eig_vec_tt
    eig_vec_tt = tt_matrix_vec_mul(matrix_tt, eig_vec_tt)
    eig_val = tt_inner_prod(prev_eig_vec, eig_vec_tt)
    return tt_normalise(eig_vec_tt), (2*normalisation-eig_val)
"""


def tt_randomised_min_eigentensor_exp(block_matrix_tt: List[np.array], block_Delta_tt, num_iter=10, tol=1e-6):
    """
    Only for symmetric matrices
    """
    n = len(block_matrix_tt)
    upper_ranks = tt_ranks(block_matrix_tt)
    target_ranks = [1] + upper_ranks + [1]
    gaussian_tt = [
        np.divide(1, l_n * 2 * l_np1) * np.random.randn(l_n, 2, l_np1)
        for i, (l_n, l_np1) in enumerate(zip(target_ranks[:-1], target_ranks[1:]))
    ]
    np.set_printoptions(linewidth=800, threshold=np.inf)
    block_matrix_tt = [np.array([[1, 1], [1, 1]]).reshape(1, 2, 2, 1)] + block_matrix_tt
    block_Delta_tt = [np.array([[-1, -0.75], [-0.5, -0.25]]).reshape(1, 2, 2, 1)] + block_Delta_tt
    block_matrix_tt = tt_rank_reduce(tt_add(block_matrix_tt, block_Delta_tt), err_bound=0)
    normalisation = np.sqrt(tt_block_inner_prod(block_matrix_tt, block_matrix_tt, 1)).reshape(1, 2, 2, 1)
    block_matrix_tt[0] *= -np.divide(1, normalisation)
    identity = [np.array([[2, 2], [2, 2]]).reshape(1, 2, 2, 1)] + tt_identity(n)
    block_matrix_tt = tt_rank_reduce(tt_add(identity, block_matrix_tt), err_bound=0)
    block_eig_vec_tt = tt_normalise([np.random.randn(1, 2, 2, 1)] + [np.random.randn(1, 2, 1) for _ in range(n)])
    norm_2 = np.inf
    for i in range(num_iter):
        prev_norm_2 = norm_2
        block_eig_vec_tt = tt_block_matrix_vec_mul(block_matrix_tt, block_eig_vec_tt, 1)
        block_eig_vec_tt = tt_rank_retraction(block_eig_vec_tt, upper_ranks)
        block_norm_2 = tt_block_inner_prod(block_eig_vec_tt, block_eig_vec_tt, 1)
        block_eig_vec_tt[0] *= np.divide(1, np.sqrt(block_norm_2[0]))
        if np.less_equal(np.abs(norm_2 - prev_norm_2), tol):
            break
    prev_block_eig_vec_tt = block_eig_vec_tt
    block_eig_vec_tt = tt_block_matrix_vec_mul(block_matrix_tt, block_eig_vec_tt, 1)
    block_eig_vec_tt = tt_rank_retraction(block_eig_vec_tt, upper_ranks)
    block_eig_vals = tt_block_inner_prod(block_eig_vec_tt, prev_block_eig_vec_tt, 1)
    return tt_normalise(block_eig_vec_tt), normalisation * (2 - block_eig_vals[0])


eig_val = np.min(np.linalg.eigvals(tt_matrix_to_matrix(copy.deepcopy(psd_matrix_tt_add))))
print("Eig val actual truth: ", eig_val)
#_, eig_val_1 = tt_min_eigentensor(copy.deepcopy(psd_matrix_tt), 500)
#print("Eig val my truth: ", eig_val_1)
_, eig_val_2 = tt_randomised_min_eigentensor_exp(copy.deepcopy(psd_matrix_tt), copy.deepcopy(Delta_tt), 500)
print("Eig val part: ", eig_val_2)
