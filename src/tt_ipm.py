import sys
import os

import numpy as np

sys.path.append(os.getcwd() + '/../')

from src.tt_ops import *
from src.tt_ops import tt_rank_reduce, tt_mask_to_linear_op
from src.tt_amen import tt_amen
from src.tt_eig import tt_min_eig
from src.tt_ineq_check import tt_is_geq, tt_is_geq_, tt_is_leq


IDX_00 = [
    np.array([[1, 0, 0, 0],
              [0, 0, 0, 0],
              [0, 0, 0, 0],
              [0, 0, 0, 0]]).reshape(1, 4, 4, 1)
]

IDX_11 = [
    np.array([[0, 0, 0, 0],
              [0, 1, 0, 0],
              [0, 0, 0, 0],
              [0, 0, 0, 0]]).reshape(1, 4, 4, 1)
]

IDX_21 = [
    np.array([[0, 0, 0, 0],
              [0, 0, 0, 0],
              [0, 1, 0, 0],
              [0, 0, 0, 0]]).reshape(1, 4, 4, 1)
]

IDX_31 = [
    np.array([[0, 0, 0, 0],
              [0, 0, 0, 0],
              [0, 0, 0, 0],
              [0, 1, 0, 0]]).reshape(1, 4, 4, 1)
]

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


def tt_infeasible_feas_rhs(
    obj_tt,
    lin_op_tt,
    lin_op_tt_adj,
    bias_tt,
    lin_op_tt_ineq,
    lin_op_tt_ineq_adj,
    bias_tt_ineq,
    X_tt,
    Y_tt,
    T_tt,
    Z_tt,
    mu,
    tol,
    active_ineq
):
    dual_feas = tt_sub(tt_add(obj_tt, Z_tt), tt_mat(tt_linear_op(lin_op_tt_adj, Y_tt), shape=(2, 2)))
    primal_feas = tt_sub(bias_tt, tt_mat(tt_linear_op(lin_op_tt, X_tt), shape=(2, 2)))
    middle_rhs = IDX_1 + primal_feas
    primal_dual_error = tt_inner_prod(dual_feas, dual_feas) + tt_inner_prod(primal_feas, primal_feas)
    if active_ineq:
        dual_feas = tt_sub(dual_feas, tt_mat(tt_linear_op(lin_op_tt_ineq_adj, T_tt), shape=(2, 2)))
        dual_feas = tt_rank_reduce(dual_feas, err_bound=(5e-5)*tol)
        primal_feas_ineq = tt_hadamard(T_tt, tt_sub(tt_mat(tt_linear_op(lin_op_tt_ineq, X_tt), shape=(2, 2)), bias_tt_ineq))
        primal_feas_ineq = tt_rank_reduce(tt_sub(primal_feas_ineq, tt_scale(mu, tt_one_matrix(len(X_tt)))), err_bound=(5e-5)*tol)
        middle_rhs = tt_add(middle_rhs, IDX_2 + primal_feas_ineq)
    upper_rhs = IDX_0 + dual_feas
    newton_rhs = tt_add(upper_rhs, middle_rhs)
    XZ_term = tt_mat_mat_mul(X_tt, Z_tt)
    lower_rhs = IDX_3 + tt_add(XZ_term, tt_transpose(XZ_term))
    newton_rhs = tt_add(newton_rhs, lower_rhs)
    newton_rhs = tt_rank_reduce(newton_rhs, err_bound=(5e-5)*tol)
    newton_rhs = tt_sub(newton_rhs, IDX_3 + tt_scale(2 * mu, tt_identity(len(X_tt))))
    return tt_scale(-1, tt_vec(newton_rhs)), primal_dual_error


def tt_infeasible_newton_system_lhs(
        lhs_skeleton,
        X_tt,
        Z_tt,
        T_tt,
        lin_op_tt_ineq,
        bias_tt_ineq,
        tol,
        active_ineq
):
    Z_op_tt = IDX_31 + tt_op_to_mat(tt_add(tt_op_left_from_tt_matrix(Z_tt), tt_op_right_from_tt_matrix(Z_tt)))
    X_op_tt = IDX_33 + tt_op_to_mat(tt_add(tt_op_right_from_tt_matrix(X_tt), tt_op_left_from_tt_matrix(X_tt)))
    newton_system = tt_add(lhs_skeleton, Z_op_tt)
    newton_system = tt_add(newton_system, X_op_tt)
    if active_ineq:
        ineq_res_tt = tt_rank_reduce(tt_sub(tt_mat(tt_linear_op(lin_op_tt_ineq, X_tt), shape=(2, 2)), bias_tt_ineq), err_bound=0.1*tol)
        ineq_res_op_tt = tt_op_to_mat(tt_mask_to_linear_op(ineq_res_tt))
        T_op_tt = tt_scale(-1, tt_mask_to_linear_op(T_tt))
        T_comp_linear_op_tt_ineq = tt_op_to_mat(tt_op_op_compose(T_op_tt, lin_op_tt_ineq))
        newton_system = tt_add(newton_system, IDX_22 + ineq_res_op_tt)
        newton_system = tt_add(newton_system, IDX_21 + T_comp_linear_op_tt_ineq)
    else:
        # For better conditioning
        newton_system = tt_add(newton_system, IDX_22 + tt_op_to_mat(tt_op_right_from_tt_matrix(tt_identity(len(X_tt)))))
    return tt_rank_reduce(newton_system, err_bound=0.1*tol)


def _tt_get_block(i, j, block_matrix_tt):
    first_core = block_matrix_tt[0][:, i, j, :]
    first_core = np.einsum("ab, bcde -> acde", first_core, block_matrix_tt[1])
    return [first_core] + block_matrix_tt[2:]

def _tt_ipm_newton_step(
        obj_tt,
        lhs_skeleton,
        lin_op_tt,
        lin_op_tt_adj,
        bias_tt,
        lin_op_tt_ineq,
        lin_op_tt_ineq_adj,
        bias_tt_ineq,
        X_tt,
        Y_tt,
        T_tt,
        Z_tt,
        tol,
        active_ineq,
        verbose
):
    mu = tt_inner_prod(Z_tt, [0.5 * c for c in X_tt])
    lhs_matrix_tt = tt_infeasible_newton_system_lhs(
        lhs_skeleton,
        X_tt,
        Z_tt,
        T_tt,
        lin_op_tt_ineq,
        bias_tt_ineq,
        tol,
        active_ineq
    )
    rhs_vec_tt, primal_dual_error = tt_infeasible_feas_rhs(
        obj_tt,
        lin_op_tt,
        lin_op_tt_adj,
        bias_tt,
        lin_op_tt_ineq,
        lin_op_tt_ineq_adj,
        bias_tt_ineq,
        X_tt,
        Y_tt,
        T_tt,
        Z_tt,
        0.5 * mu,
        tol,
        active_ineq
    )
    Delta_tt, res = tt_amen(lhs_matrix_tt, rhs_vec_tt, kickrank=4, verbose=verbose)
    Delta_tt = tt_mat(Delta_tt, shape=(2, 2))
    Delta_Y_tt = tt_rank_reduce(_tt_get_block(0, 0, Delta_tt), err_bound=tol)
    Delta_X_tt = tt_rank_reduce(_tt_get_block(0, 1, Delta_tt), err_bound=tol)
    Delta_T_tt = tt_rank_reduce(_tt_get_block(1, 0, Delta_tt), err_bound=tol) if active_ineq else None
    Delta_Z_tt = tt_rank_reduce(_tt_get_block(1, 1, Delta_tt), err_bound=tol)
    x_step_size, z_step_size = _tt_line_search(X_tt, T_tt, Z_tt, Delta_X_tt, Delta_T_tt, Delta_Z_tt, lin_op_tt_ineq, bias_tt_ineq, active_ineq)
    X_tt = tt_rank_reduce(tt_add(X_tt, tt_scale(0.98 * x_step_size, Delta_X_tt)), err_bound=0.1*tol)
    Y_tt = tt_rank_reduce(tt_add(Y_tt, tt_scale(0.98 * z_step_size, Delta_Y_tt)), err_bound=tol)
    Z_tt = tt_rank_reduce(tt_add(Z_tt, tt_scale(0.98 * z_step_size, Delta_Z_tt)), err_bound=0.1*tol)
    if active_ineq:
        T_tt = tt_rank_reduce(tt_add(T_tt, tt_scale(0.98 * z_step_size, Delta_T_tt)), err_bound=tol)

    if verbose:
        print(f"Step sizes: {x_step_size}, {z_step_size}")

    # TODO: Mind the error bound here! May lead to inaccuracies and more iterations
    return X_tt, Y_tt, T_tt, Z_tt, primal_dual_error, mu


def _tt_line_search(
        X_tt,
        T_tt,
        Z_tt,
        Delta_X_tt,
        Delta_T_tt,
        Delta_Z_tt,
        lin_op_tt_ineq,
        bias_tt_ineq,
        active_ineq,
        iters=15,
        crit=1e-12
):
    x_step_size = 1
    z_step_size = 1
    discount = 0.5
    discount_x = False
    discount_z = False
    r = X_tt[0].shape[-1]
    new_X_tt = tt_add(X_tt, Delta_X_tt)
    for iter in range(iters):
        val_x, _, _ = tt_min_eig(new_X_tt)
        discount_x = np.greater(val_x, crit)
        if discount_x:
            break
        else:
            new_X_tt[0][:, :, :, r:] *= discount
            x_step_size *= discount

    if active_ineq:
        for iter in range(iters):
            discount_x, val, _ = tt_is_geq(lin_op_tt_ineq, new_X_tt, bias_tt_ineq, crit=crit)
            if discount_x:
                break
            else:
                new_X_tt[0][:, :, :, r:] *= discount
                x_step_size *= discount
    new_Z_tt = tt_add(Z_tt, Delta_Z_tt)
    r = Z_tt[0].shape[-1]
    for iter in range(15):
        val_z, _, _ = tt_min_eig(new_Z_tt)
        discount_z = np.greater(val_z, crit)
        if discount_z:
            break
        else:
            new_Z_tt[0][:, :, :, r:] *= discount
            z_step_size *= discount

    if active_ineq:
        new_T_tt = tt_add(T_tt, tt_scale(z_step_size, Delta_T_tt))
        for iter in range(iters):
            discount_z, val, _ = tt_is_geq_(new_T_tt, crit=crit)
            if discount_z:
                break
            else:
                new_T_tt[0][:, :, :, r:] *= discount
                z_step_size *= discount
    return discount_x*x_step_size, discount_z*z_step_size


def tt_ipm(
    obj_tt,
    lin_op_tt,
    lin_op_tt_adj,
    bias_tt,
    lin_op_tt_ineq=None,
    lin_op_tt_ineq_adj=None,
    bias_tt_ineq=None,
    max_iter=100,
    feasibility_tol=1e-4,
    centrality_tol=1e-3,
    verbose=False
):
    dim = len(obj_tt)
    active_ineq = lin_op_tt_ineq is not None and lin_op_tt_ineq_adj is not None and bias_tt_ineq is not None
    op_tt_adjoint = IDX_00 + tt_op_to_mat(tt_scale(-1, lin_op_tt_adj))
    op_tt = IDX_11 + tt_op_to_mat(tt_scale(-1, lin_op_tt))
    I_mat_tt = tt_op_to_mat(tt_op_right_from_tt_matrix(tt_identity(dim)))
    I_op_tt = IDX_03 + I_mat_tt
    lhs_skeleton = tt_add(op_tt, op_tt_adjoint)
    lhs_skeleton = tt_add(lhs_skeleton, I_op_tt)
    if active_ineq:
        op_tt_ineq_adjoint = IDX_02 + tt_op_to_mat(tt_scale(-1, lin_op_tt_ineq_adj))
        lhs_skeleton = tt_add(lhs_skeleton, op_tt_ineq_adjoint)
    lhs_skeleton = tt_rank_reduce(lhs_skeleton, err_bound=0.1*feasibility_tol)
    X_tt = tt_identity(dim)
    Y_tt = tt_zero_matrix(dim)
    T_tt = tt_one_matrix(dim)
    Z_tt = tt_identity(dim)
    iter = 0
    feasible = False
    for iter in range(max_iter):
        X_tt, Y_tt, T_tt, Z_tt, pd_error, mu = _tt_ipm_newton_step(
            obj_tt,
            lhs_skeleton,
            lin_op_tt,
            lin_op_tt_adj,
            bias_tt,
            lin_op_tt_ineq,
            lin_op_tt_ineq_adj,
            bias_tt_ineq,
            X_tt,
            Y_tt,
            T_tt,
            Z_tt,
            feasibility_tol,
            active_ineq,
            verbose
        )
        if verbose:
            print(f"---Step {iter}---")
            print(f"Duality Gap: {100 * np.abs(mu):.4f}%")
            print(f"Primal-Dual error: {pd_error:.8f}")
            print(
                f"Ranks X_tt: {tt_ranks(X_tt)}, Z_tt: {tt_ranks(Z_tt)}, \n"
                f"      Y_tt: {tt_ranks(Y_tt)}, T_tt: {tt_ranks(T_tt)} \n"
            )

        if np.less(pd_error, feasibility_tol):
            if not feasible and verbose:
                print("-------------------------")
                print(f"IPM reached feasibility!")
                print("-------------------------")
            feasible = True
            if np.less(np.abs(mu), centrality_tol):
                break
    if verbose:
        print(f"Converged in {iter + 1} iterations.")
    return X_tt, Y_tt, T_tt, Z_tt
