import sys
import os

sys.path.append(os.getcwd() + '/../')

import numpy as np
import copy
import time
from typing import List
from src.tt_ops import *
from src.tt_ops import tt_rank_reduce, _tt_mat_core_collapse, _block_diag_tensor, tt_mask_to_linear_op
from src.tt_amen import tt_amen

IDX_01 = [
    np.array([[0, 1, 0, 0],
              [0, 0, 0, 0],
              [0, 0, 0, 0],
              [0, 0, 0, 0]]).reshape(1, 4, 4, 1)
]
IDX_10 = [
    np.array([[0, 0, 0, 0],
              [1, 0, 0, 0],
              [0, 0, 0, 0],
              [0, 0, 0, 0]]).reshape(1, 4, 4, 1)
]
IDX_02 = [
    np.array([[0, 0, 1, 0],
              [0, 0, 0, 0],
              [0, 0, 0, 0],
              [0, 0, 0, 0]]).reshape(1, 4, 4, 1)
]
IDX_20 = [
    np.array([[0, 0, 0, 0],
              [0, 0, 0, 0],
              [1, 0, 0, 0],
              [0, 0, 0, 0]]).reshape(1, 4, 4, 1)
]
IDX_22 = [
    np.array([[0, 0, 0, 0],
              [0, 0, 0, 0],
              [0, 0, 1, 0],
              [0, 0, 0, 0]]).reshape(1, 4, 4, 1)
]
IDX_33 = [
    np.array([[0, 0, 0, 0],
              [0, 0, 0, 0],
              [0, 0, 0, 0],
              [0, 0, 0, 1]]).reshape(1, 4, 4, 1)
]

IDX_0 = [np.array([[1, 0], [0, 0]]).reshape(1, 2, 2, 1)]
IDX_1 = [np.array([[0, 1], [0, 0]]).reshape(1, 2, 2, 1)]
IDX_2 = [np.array([[0, 0], [1, 0]]).reshape(1, 2, 2, 1)]


def tt_infeasible_newton_system_rhs(obj_tt, linear_op_tt, bias_tt, X_tt, Y_tt, Z_tt, mu):
    upper_rhs = IDX_0 + tt_sub(tt_add(Z_tt, obj_tt), tt_mat(tt_linear_op(tt_adjoint(linear_op_tt), Y_tt), shape=(2, 2)))
    middle_rhs = IDX_1 + tt_sub(bias_tt, tt_mat(tt_linear_op(linear_op_tt, X_tt), shape=(2, 2)))
    lower_rhs = IDX_2 + tt_sub(tt_mat_mat_mul(Z_tt, X_tt), tt_scale(mu, tt_identity(len(X_tt))))
    newton_rhs = tt_add(upper_rhs, middle_rhs)
    newton_rhs = tt_add(newton_rhs, lower_rhs)
    return tt_rank_reduce(tt_vec(newton_rhs))


def tt_infeasible_newton_system_lhs(linear_op_tt, X_tt, Z_tt):
    linear_op_tt = tt_scale(-1, linear_op_tt)
    linear_op_tt_adjoint = IDX_01 + tt_op_to_mat(tt_adjoint(linear_op_tt))
    linear_op_tt = IDX_10 + tt_op_to_mat(linear_op_tt)
    I_mat_tt = tt_op_to_mat(tt_op_from_tt_matrix(tt_identity(len(X_tt))))
    I_op_tt = IDX_02 + I_mat_tt
    Z_op_tt = IDX_20 + tt_op_to_mat(tt_op_from_tt_matrix(Z_tt))
    X_op_tt = IDX_22 + tt_op_to_mat(tt_op_from_tt_matrix(X_tt))
    newton_system = tt_add(linear_op_tt, linear_op_tt_adjoint)
    newton_system = tt_add(newton_system, I_op_tt)
    newton_system = tt_add(newton_system, Z_op_tt)
    newton_system = tt_add(newton_system, X_op_tt)
    # For numerical stability?
    newton_system = tt_add(newton_system, IDX_33 + I_mat_tt)
    return tt_rank_reduce(newton_system)


def _tt_get_block(i, j, block_matrix_tt):
    first_core = block_matrix_tt[0][:, i, j, :]
    first_core = np.einsum("ab, bcde -> acde", first_core, block_matrix_tt[1])
    return [first_core] + block_matrix_tt[2:]


def _tt_ipm_newton_step(obj_tt, linear_op_tt, bias_tt, XZ_tt, Y_tt, beta=5e-3):
    X_tt = _tt_get_block(0, 0, XZ_tt)
    Z_tt = _tt_get_block(0, 1, XZ_tt)
    lhs_matrix_tt = tt_infeasible_newton_system_lhs(linear_op_tt, X_tt, Z_tt)
    mu = tt_inner_prod(Z_tt, [0.5 * c for c in X_tt])
    rhs_vec_tt = tt_rank_reduce(tt_infeasible_newton_system_rhs(obj_tt, linear_op_tt, bias_tt, X_tt, Y_tt, Z_tt, mu),
                                err_bound=0.5 * beta)
    #np.set_printoptions(threshold=np.inf, linewidth=200)
    #lhs_matrix_tt_2 = tt_rank_reduce(tt_mat_mat_mul(tt_transpose(lhs_matrix_tt), lhs_matrix_tt))
    #rhs_vec_tt_2 = tt_rank_reduce(tt_matrix_vec_mul(tt_transpose(lhs_matrix_tt), rhs_vec_tt))
    #print(tt_ranks(lhs_matrix_tt_2), tt_ranks(rhs_vec_tt_2))
    # Tikhononv regularization
    lhs_matrix_tt_reg = tt_rank_reduce(
        tt_add(lhs_matrix_tt, [beta * np.eye(4).reshape(1, 4, 4, 1) for _ in range(len(lhs_matrix_tt))]),
        err_bound=0.5 * beta)
    Delta_tt = tt_amen(lhs_matrix_tt_reg, rhs_vec_tt, verbose=True, nswp=10)
    res = tt_sub(tt_matrix_vec_mul(lhs_matrix_tt, Delta_tt), rhs_vec_tt)
    print("Error: ", np.sqrt(tt_inner_prod(res, res)))
    return Delta_tt


def _tt_homotopy_step(XZ_tt, Delta_XZ_tt):
    XZ_tt = tt_rank_reduce(tt_add(XZ_tt, Delta_XZ_tt))
    # Projection to PSD-cone
    V_tt = tt_burer_monteiro_factorisation(XZ_tt)
    return V_tt


def _symmetrisation(Delta_tt):
    return tt_rank_reduce(tt_scale(0.5, tt_add(Delta_tt, tt_transpose(Delta_tt))))


def _get_xz_block(XYZ_tt):
    new_index_block = np.zeros_like(XYZ_tt[0])
    new_index_block[:, 0] += XYZ_tt[0][:, 0]
    return new_index_block + XYZ_tt[1:]


def tt_ipm(obj_tt, linear_op_tt, bias_tt):
    dim = len(obj_tt)
    V_tt = [np.array([[1, 1], [0, 0]]).reshape(1, 2, 2, 1)] + tt_identity(dim)  # [X, 0, Z, 0]^T
    Y_tt = tt_zeros(dim, shape=(2, 2))
    for _ in range(1):
        XZ_tt = tt_rank_reduce(tt_mat_mat_mul(V_tt, tt_transpose(V_tt)))
        Delta_tt = _tt_ipm_newton_step(obj_tt, linear_op_tt, bias_tt, XZ_tt, Y_tt)
        #Delta_tt = _symmetrisation(Delta_tt)
        #Delta_Y_tt = _tt_get_block(1, 0, Delta_tt)
        #Delta_XZ_tt = _get_xz_block(Delta_tt)
        #V_tt = _tt_homotopy_step(XZ_tt, Delta_XZ_tt)
        ## TODO: get step size that minimises dual residual
        #Y_tt = tt_rank_reduce(tt_add(Y_tt, Delta_Y_tt))
    return V_tt, Y_tt


if __name__ == "__main__":
    np.random.seed(84)
    random_M = tt_random_gaussian([2], shape=(2, 2))
    linear_op_tt = tt_rank_reduce(tt_mask_to_linear_op(tt_add(random_M, tt_transpose(random_M))))
    initial_guess = tt_random_gaussian([2], shape=(2, 2))
    initial_guess = tt_rank_reduce(tt_mat_mat_mul(initial_guess, tt_transpose(initial_guess)))
    bias_tt = tt_rank_reduce(tt_mat(tt_linear_op(linear_op_tt, initial_guess), shape=(2, 2)))
    obj_tt = tt_identity(len(bias_tt))
    _ = tt_ipm(obj_tt, linear_op_tt, bias_tt)
