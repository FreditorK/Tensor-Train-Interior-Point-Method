import copy
import sys
import os

import numpy as np
import scipy.linalg
import scipy.optimize
import scipy.sparse.linalg

sys.path.append(os.getcwd() + '/../')

from src.tt_ops import *
from src.tt_ops import tt_rank_reduce
from src.tt_amen import tt_block_amen,  svd_solve_local_system
from src.tt_eig import tt_min_eig
from src.tt_ineq_check import tt_is_geq, tt_is_geq_, tt_is_psd


def vec(matrix):
    return np.reshape(matrix, (-1, 1), order="F")

def mat(vector):
    dim = int(np.sqrt(vector))
    return np.reshape(vector, (dim, dim), order="F")

def tt_infeasible_newton_system(
        lhs_skeleton,
        vec_obj,
        X,
        vec_Y,
        Z,
        T,
        mat_lin_op,
        mat_lin_op_adj,
        vec_bias,
        mat_lin_op_ineq,
        mat_lin_op_ineq_adj,
        vec_bias_ineq,
        mu,
        tol,
        feasibility_tol,
        eps,
        active_ineq
):
    idx_add = int(active_ineq)
    # TODO: scaling matrix needs to be full rank, otherwise we have problems
    scaling_matrix = np.eye(len(Z))
    L_Z = np.kron(scaling_matrix, Z) + np.kron(Z, scaling_matrix)
    L_X = np.kron(X, scaling_matrix) + np.kron(scaling_matrix, X)
    block_dim = len(L_Z)
    vec_X = vec(X)

    if active_ineq:
        ineq_res = vec_bias_ineq - mat_lin_op_ineq @ vec_X
        mat_ineq_res_op = np.diag(ineq_res)
        mat_T_op = np.diag(vec(T))
        lhs_skeleton[2*block_dim:3*block_dim, 2*block_dim:3*block_dim] = mat_ineq_res_op
        lhs_skeleton[2*block_dim:3*block_dim, :block_dim] = -mat_T_op @ mat_lin_op_ineq
    lhs_skeleton[(2 + idx_add)*block_dim:, :block_dim] = L_Z
    lhs_skeleton[(2 + idx_add)*block_dim:, (2 + idx_add)*block_dim:] = L_X

    rhs = np.zeros((len(lhs_skeleton), 1))
    dual_feas = mat_lin_op_adj @ vec_Y - vec(Z) - vec_obj
    primal_feas = mat_lin_op @ vec_X - vec_bias  # primal feasibility
    primal_error = tt_inner_prod(primal_feas, primal_feas)
    if primal_error > feasibility_tol:
        rhs[block_dim:2*block_dim] = primal_feas

    if active_ineq:
        vec_T = vec(T)
        dual_feas = dual_feas + mat_lin_op_ineq_adj @ vec_T
        primal_feas_ineq = mat_lin_op_ineq @ vec_X - vec_bias_ineq
        # TODO: Does mu 1 not also be under mat_lin_op_tt_ineq, need to adjust mu 1 to have zeros where L(X) has zeros
        one = mat_lin_op_ineq_adj @ np.ones_like(vec_X)
        one = np.divide(one, vec_T)
        primal_feas_ineq = primal_feas_ineq + mu*one
        primal_ineq_error = np.trace(primal_feas_ineq.T @ primal_feas_ineq)
        if primal_ineq_error > tol:
            rhs[2*block_dim:3*block_dim] = primal_feas_ineq
            primal_error += primal_ineq_error

    dual_error = np.trace(dual_feas.T @ dual_feas)
    XZ_term = L_Z @ vec_X
    rhs[(2 + idx_add)*block_dim] = 2*mu*vec(np.eye(len(X))) - XZ_term
    if dual_error > feasibility_tol:
        rhs[:block_dim] = dual_feas

    return lhs_skeleton, rhs, primal_error + dual_error


def _symmetrise(matrix):
    return (matrix.T + matrix)/2


def _ipm_newton_step(
        lag_maps,
        vec_obj,
        lhs_skeleton,
        mat_lin_op,
        mat_lin_op_adj,
        vec_bias,
        mat_lin_op_ineq,
        mat_lin_op_ineq_adj,
        vec_bias_ineq,
        X,
        vec_Y,
        T,
        Z,
        tol,
        feasibility_tol,
        active_ineq,
        verbose,
        eps,
        sigma
):
    mu = np.trace(Z.T @ X)/(2**(len(Z)))
    lhs_matrix, rhs_vec, primal_dual_error = tt_infeasible_newton_system(
        lhs_skeleton,
        vec_obj,
        X,
        vec_Y,
        Z,
        T,
        mat_lin_op,
        mat_lin_op_adj,
        vec_bias,
        mat_lin_op_ineq,
        mat_lin_op_ineq_adj,
        vec_bias_ineq,
        max(sigma * mu,  1e-3),
        tol,
        feasibility_tol,
        eps,
        active_ineq
    )
    idx_add = int(active_ineq)
    Delta = scipy.linalg.solve(lhs_matrix, rhs_vec, check_finite=False)
    block_dim = len(Delta) // 4
    vec_Delta_Y = Delta[block_dim:2*block_dim]
    Delta_T = mat(Delta[2*block_dim:3*block_dim]) if active_ineq else None
    Delta_X = mat(Delta[:block_dim])
    Delta_Z = Delta[(2 + idx_add)*block_dim:]
    Delta_X = _symmetrise(Delta_X)
    Delta_Z = _symmetrise(Delta_Z)
    x_step_size, z_step_size = _line_search(X, T, Z, Delta_X, Delta_T, Delta_Z, mat_lin_op_ineq, vec_bias_ineq, active_ineq, crit=0.1 * tol)
    X = X + 0.98 * x_step_size*Delta_X
    vec_Y = vec_Y + 0.98 * z_step_size*vec_Delta_Y
    Z = Z + 0.98 * z_step_size*Delta_Z
    if active_ineq:
        # FIXME: Note that T_tt should grow large on the zeros of b - L_ineq(X_tt)
        T = T + 0.98 * z_step_size*Delta_T

    if verbose:
        print(f"Step sizes: {x_step_size}, {z_step_size}")

    #print("Report ---")
    #print("Delta Y")
    #print(np.round(tt_matrix_to_matrix(tt_mat(vec_Delta_Y_tt)), decimals=3))
    #if active_ineq:
    #    print("Delta T")
    #    print(np.round(tt_matrix_to_matrix(Delta_T_tt), decimals=3))
    #print("Delta X")
    #print(np.round(tt_matrix_to_matrix(Delta_X_tt), decimals=3))
    #print("Delta Z")
    #print(np.round(tt_matrix_to_matrix(Delta_Z_tt), decimals=3))

    return X, vec_Y, T, Z, primal_dual_error, mu


def _line_search(
        X,
        T,
        Z,
        Delta_X,
        Delta_T,
        Delta_Z,
        lin_op_ineq,
        vec_bias_ineq,
        active_ineq,
        iters=15
):
    x_step_size = 1
    z_step_size = 1
    discount = 0.5
    discount_x = False
    discount_z = False

    for iter in range(iters):
        discount_x = np.min(np.linalg.eigvalsh(X + discount_x * Delta_X)) > 0
        if discount_x:
            break
        else:
            x_step_size *= discount
    if active_ineq and discount_x:
        for iter in range(iters):
            discount_x = vec_bias_ineq - lin_op_ineq @ (X + discount_x * Delta_X) > 0
            if discount_x:
                break
            else:
                x_step_size *= discount

    for iter in range(iters):
        discount_z = np.min(np.linalg.eigvalsh(Z + discount_z * Delta_Z)) > 0
        if discount_z:
            break
        else:
            z_step_size *= discount
    if active_ineq and discount_z:
        for iter in range(iters):
            discount_z = np.min(T + discount_z*Delta_T) > 0
            if discount_z:
                break
            else:
                z_step_size *= discount
    return discount_x*x_step_size, discount_z*z_step_size


def ipm(
    lag_maps,
    obj_tt,
    lin_op_tt,
    bias_tt,
    lin_op_tt_ineq=None,
    bias_tt_ineq=None,
    max_iter=100,
    feasibility_tol=1e-5,
    centrality_tol=1e-2,
    verbose=False,
    eps=1e-10
):
    dim = len(obj_tt)
    feasibility_tol = feasibility_tol / np.sqrt(dim)
    centrality_tol = centrality_tol / np.sqrt(dim)
    op_tol = 0.5*min(feasibility_tol, centrality_tol)
    lag_maps = {key: tt_matrix_to_matrix(value) for key, value in lag_maps.items()}
    obj = tt_matrix_to_matrix(obj_tt)
    lin_op = tt_matrix_to_matrix(lin_op_tt)
    bias = tt_matrix_to_matrix(bias_tt)

    active_ineq = lin_op_tt_ineq is not None or bias_tt_ineq is not None
    block_size = len(obj_tt)
    lhs_skeleton = np.zeros(4*block_size, 4*block_size)
    lin_op_adj = lin_op.T
    lhs_skeleton[:block_size, block_size:2*block_size] = -lin_op_adj
    lhs_skeleton[block_size:2*block_size, :block_size] = -lin_op
    lin_op_tt_ineq_adj = None
    if active_ineq:
        bias_ineq = tt_matrix_to_matrix(bias_tt_ineq)
        lin_op_ineq = tt_matrix_to_matrix(lin_op_tt_ineq)
        lin_op_ineq_adj = -lin_op_ineq.T
        lhs_skeleton[:block_size, 2*block_size:3*block_size] = -lin_op_ineq_adj
        lhs_skeleton[:block_size, 3*block_size:4*block_size] = np.eye(2**(2*dim))
        bias_ineq = vec(bias_ineq)
    else:
        lhs_skeleton[:block_size, 2*block_size:3*block_size] = np.eye(2**(2*dim))
    X = np.eye(2**dim)
    vec_Y = np.zeros((2**dim, 2**dim))
    T = np.ones((2**dim, 2**dim))
    if active_ineq:
        T = mat(lin_op_tt_ineq_adj @ vec(T))
    Z = np.eye(2**dim)
    iter = 0
    sigma = 0.5
    for iter in range(1, max_iter):
        X, vec_Y, T, Z, pd_error, mu = _ipm_newton_step(
            lag_maps,
            obj,
            lhs_skeleton,
            lin_op,
            lin_op_adj,
            bias,
            lin_op_ineq,
            lin_op_ineq_adj,
            bias_ineq,
            X,
            vec_Y,
            T,
            Z,
            op_tol,
            feasibility_tol,
            active_ineq,
            verbose,
            eps,
            sigma
        )
        sigma *= max(min((pd_error / mu)**2, 1), 0.01)
        if verbose:
            print(f"---Step {iter}---")
            print(f"Duality Gap: {100 * np.abs(mu):.4f}%")
            print(f"Primal-Dual error: {pd_error:.8f}")
            print(f"Sigma: {sigma:.4f}")

        if np.less(pd_error, feasibility_tol) and np.less(np.abs(mu), centrality_tol):
                break
    if verbose:
        print(f"---Terminated---")
        print(f"Converged in {iter} iterations.")
    return X, mat(vec_Y), T, Z
