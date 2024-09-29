import sys
import os

import numpy as np
from numpy.core.defchararray import lower
from scipy.optimize import newton

sys.path.append(os.getcwd() + '/../')

from src.tt_ops import *
from src.tt_ops import tt_rank_reduce, tt_mask_to_linear_op
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
IDX_03 = [
    np.array([[0, 0, 0, 1],
              [0, 0, 0, 0],
              [0, 0, 0, 0],
              [0, 0, 0, 0]]).reshape(1, 4, 4, 1)
]
IDX_30 = [
    np.array([[0, 0, 0, 0],
              [0, 0, 0, 0],
              [0, 0, 0, 0],
              [1, 0, 0, 0]]).reshape(1, 4, 4, 1)
]
IDX_33 = [
    np.array([[0, 0, 0, 0],
              [0, 0, 0, 0],
              [0, 0, 0, 0],
              [0, 0, 0, 1]]).reshape(1, 4, 4, 1)
]

IDX_0 = [np.array([[1, 0],
                   [0, 0]]).reshape(1, 2, 2, 1)]
IDX_1 = [np.array([[0, 1],
                   [0, 0]]).reshape(1, 2, 2, 1)]
IDX_2 = [np.array([[0, 0],
                   [1, 0]]).reshape(1, 2, 2, 1)]
IDX_3 = [np.array([[0, 0],
                   [0, 1]]).reshape(1, 2, 2, 1)]


def tt_infeasible_newton_system_rhs(
    obj_tt,
    linear_op_tt,
    bias_tt,
    X_tt,
    Y_tt,
    Z_tt,
    Delta_XZ_tt,
    mu,
    feasible
):
    # Mehrotra's Aggregated System
    if feasible:
        # If primal-dual error is beneath tolerance, quicken up newton_system construction by assuming zero pd-error
        newton_rhs = IDX_3 + tt_sub(tt_mat_mat_mul(X_tt, Z_tt), tt_scale(mu, tt_identity(len(X_tt))))
        primal_dual_error = 0
    else:
        upper_rhs = IDX_0 + tt_sub(tt_add(obj_tt, Z_tt),
                                   tt_mat(tt_linear_op(tt_adjoint(linear_op_tt), Y_tt), shape=(2, 2)))
        middle_rhs = IDX_1 + tt_sub(bias_tt, tt_mat(tt_linear_op(linear_op_tt, X_tt), shape=(2, 2)))
        XZ_term = tt_mat_mat_mul(X_tt, Z_tt)
        Delta_X_tt = _tt_get_block(0, 0, Delta_XZ_tt)
        Delta_Z_tt = _tt_get_block(1, 1, Delta_XZ_tt)
        XZ_term = tt_add(tt_mat_mat_mul(Delta_X_tt, Delta_Z_tt), XZ_term)
        lower_rhs = IDX_3 + tt_sub(XZ_term, tt_scale(mu, tt_identity(len(X_tt))))
        newton_rhs = tt_add(upper_rhs, middle_rhs)
        primal_dual_error = tt_inner_prod(newton_rhs, newton_rhs)
        newton_rhs = tt_add(newton_rhs, lower_rhs)
    return tt_rank_reduce(tt_scale(-1, tt_vec(newton_rhs)), err_bound=0), primal_dual_error


def tt_infeasible_newton_system_lhs(lhs_skeleton, X_tt, Z_tt):
    Z_op_tt = IDX_30 + tt_op_to_mat(tt_op_from_tt_matrix(Z_tt))
    X_op_tt = IDX_33 + tt_op_to_mat(tt_op_from_tt_matrix(X_tt))
    newton_system = tt_add(lhs_skeleton, Z_op_tt)
    newton_system = tt_add(newton_system, X_op_tt)
    return tt_rank_reduce(newton_system, err_bound=0)


def _tt_get_block(i, j, block_matrix_tt):
    first_core = block_matrix_tt[0][:, i, j, :]
    first_core = np.einsum("ab, bcde -> acde", first_core, block_matrix_tt[1])
    return [first_core] + block_matrix_tt[2:]


def _tt_ipm_newton_step(
    obj_tt,
    linear_op_tt,
    lhs_skeleton,
    bias_tt,
    XZ_tt,
    Y_tt,
    Delta_XZ_tt,
    centering_param,
    feasible,
    verbose
):
    X_tt = _tt_get_block(0, 0, XZ_tt)
    Z_tt = _tt_get_block(1, 1, XZ_tt)
    mu = tt_inner_prod(Z_tt, [0.5 * c for c in X_tt])
    rhs_vec_tt, primal_dual_error = tt_infeasible_newton_system_rhs(
        obj_tt, linear_op_tt, bias_tt, X_tt, Y_tt, Z_tt, Delta_XZ_tt, centering_param * mu, feasible
    )
    lhs_matrix_tt = tt_infeasible_newton_system_lhs(lhs_skeleton, X_tt, Z_tt)
    Delta_tt, res = tt_amen(lhs_matrix_tt, rhs_vec_tt, verbose=verbose, nswp=22)
    return tt_mat(Delta_tt, shape=(2, 2)), primal_dual_error, mu


def _tt_psd_step(
    XZ_tt,
    Delta_XZ_tt,
    step_size=1,
    discount=0.5,
    max_num_steps=10,
    tol=1e-6
):
    x_step_size = step_size
    z_step_size = step_size
    discount_x = True
    discount_z = True
    # Projection to PSD-cone
    for iter in range(max_num_steps):
        new_XZ_tt = tt_add(XZ_tt, Delta_XZ_tt)
        #discount_x, discount_z = tt_block_psd(new_XZ_tt, 1000, tol)
        X_tt = _tt_get_block(0, 0, new_XZ_tt)
        Z_tt = _tt_get_block(1, 1, new_XZ_tt)
        #print("Check x: ", discount_x, np.linalg.eigvals(tt_matrix_to_matrix(X_tt)))
        #print("Check z: ", discount_z, np.linalg.eigvals(tt_matrix_to_matrix(Z_tt)))
        #print("Eigs_1: ", np.linalg.eigvals(tt_matrix_to_matrix(X_tt)))
        #print("Eigs_2: ", np.linalg.eigvals(tt_matrix_to_matrix(Z_tt)))
        discount_x = np.all(np.linalg.eigvals(tt_matrix_to_matrix(X_tt)) >= 0)
        discount_z = np.all(np.linalg.eigvals(tt_matrix_to_matrix(Z_tt)) >= 0)
        if ~discount_x:
            x_step_size *= discount
            Delta_XZ_tt[0][:, 0, 0, :] *= discount
        if ~discount_z:
            z_step_size *= discount
            Delta_XZ_tt[0][:, 1, 1, :] *= discount
        if discount_x and discount_z:
            Delta_XZ_tt[0] *= 0.95
            print(f"Step Size: {x_step_size}, {z_step_size}")
            return tt_rank_reduce(tt_add(XZ_tt, Delta_XZ_tt), err_bound=0), z_step_size
    x_step_size *= discount_x
    z_step_size *= discount_z
    Delta_XZ_tt[0][:, 0, 0, :] *= discount_x
    Delta_XZ_tt[0][:, 1, 1, :] *= discount_z
    Delta_XZ_tt[0] *= 0.95
    print(f"Step Size: {x_step_size}, {z_step_size}")
    return tt_rank_reduce(tt_add(XZ_tt, Delta_XZ_tt), err_bound=0), z_step_size


def _symmetrisation(Delta_XZ_tt):
    Delta_XZ_tt = tt_rank_reduce(tt_scale(0.5, tt_add(Delta_XZ_tt, tt_transpose(Delta_XZ_tt))), err_bound=0)
    return Delta_XZ_tt


def _get_xz_block(XYZ_tt):
    new_index_block = np.zeros_like(XYZ_tt[0])
    new_index_block[:, 0, 0] += XYZ_tt[0][:, 0, 0]
    new_index_block[:, 1, 1] += XYZ_tt[0][:, 1, 1]
    return [new_index_block] + XYZ_tt[1:]


def tt_ipm(
    obj_tt,
    linear_op_tt,
    bias_tt,
    max_iter,
    tikhonov_param=1e-4,
    interpol_damp=0.8,
    feasibility_tol=1e-8,
    centrality_tol=1e-4,
    verbose=False
):
    dim = len(obj_tt)
    tikhonov_param = tikhonov_param ** (1 / (dim - 1))
    op_tt = tt_scale(-1, linear_op_tt)
    op_tt_adjoint = IDX_01 + tt_op_to_mat(tt_adjoint(op_tt))
    op_tt = IDX_10 + tt_op_to_mat(op_tt)
    I_mat_tt = tt_op_to_mat(tt_op_from_tt_matrix(tt_identity(dim)))
    I_op_tt = IDX_03 + I_mat_tt
    lhs_skeleton = tt_add(op_tt, op_tt_adjoint)
    lhs_skeleton = tt_add(lhs_skeleton, I_op_tt)
    # Tikhononv regularization
    lhs_skeleton = tt_rank_reduce(
        tt_add(lhs_skeleton, [tikhonov_param * np.eye(4).reshape(1, 4, 4, 1) for _ in range(len(lhs_skeleton))]), err_bound=0)
    XZ_tt = tt_identity(dim + 1)  # [X, 0, 0, Z]^T
    Y_tt = tt_zeros(dim, shape=(2, 2))  # [0, Y_1, Y_2, 0]^T
    iter = 0
    centering_param = 0.5
    Delta_XZ_tt = tt_zeros(dim+1, shape=(2, 2))
    feasible = False
    for iter in range(max_iter):
        Delta_tt, pd_error, mu = _tt_ipm_newton_step(
            obj_tt, linear_op_tt, lhs_skeleton, bias_tt, XZ_tt, Y_tt, Delta_XZ_tt, centering_param, feasible, verbose
        )
        if verbose:
            print(f"---Step {iter}---")
            print("Centering Param: ", centering_param)
            print(f"Duality Gap: {abs(mu):.4f}%")
            print(f"Primal-Dual error: {pd_error:.4f}")
        condition = min(1, (pd_error / mu) ** 3)
        centering_param = interpol_damp * centering_param + (1-interpol_damp) * condition
        Delta_XZ_tt = _get_xz_block(Delta_tt)
        Delta_XZ_tt = _symmetrisation(Delta_XZ_tt)
        XZ_tt, alpha = _tt_psd_step(XZ_tt, Delta_XZ_tt)
        if np.less(pd_error, feasibility_tol):
            feasible = True
            if np.less(mu, centrality_tol):
                break
        Delta_Y_tt = _tt_get_block(0, 1, Delta_tt)
        Y_tt = tt_rank_reduce(tt_add(Y_tt, tt_scale(0.95 * alpha, Delta_Y_tt)), err_bound=0)
        #print("Y_tt:")
        #print(np.round(tt_matrix_to_matrix(Y_tt), decimals=4))
    if verbose:
        print(f"Converged in {iter + 1} iterations.")
    return XZ_tt, Y_tt


if __name__ == "__main__":
    np.random.seed(84)
    random_M = tt_random_gaussian([2], shape=(2, 2))
    linear_op_tt = tt_rank_reduce(tt_mask_to_linear_op(tt_add(random_M, tt_transpose(random_M))))
    initial_guess = tt_random_gaussian([2], shape=(2, 2))
    initial_guess = tt_rank_reduce(tt_mat_mat_mul(initial_guess, tt_transpose(initial_guess)))
    bias_tt = tt_rank_reduce(tt_mat(tt_linear_op(linear_op_tt, initial_guess), shape=(2, 2)))
    obj_tt = tt_identity(len(bias_tt))
    _ = tt_ipm(obj_tt, linear_op_tt, bias_tt)
