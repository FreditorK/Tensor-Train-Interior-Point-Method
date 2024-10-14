import sys
import os
import time

import numpy as np

from src.tt_eig import tt_min_eig

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


def tt_infeasible_centr_rhs(
    X_tt,
    Z_tt,
    mu
):
    # TODO: Loses symmetry here somehow ????
    XZ_term = tt_mat_mat_mul(X_tt, Z_tt)
    newton_rhs = IDX_3 + tt_sub(tt_add(XZ_term, tt_transpose(XZ_term)), tt_scale(2*mu, tt_identity(len(X_tt))))
    return tt_rank_reduce(tt_scale(-1, tt_vec(newton_rhs)), err_bound=0)


def tt_infeasible_feas_rhs(
    obj_tt,
    linear_op_tt,
    bias_tt,
    X_tt,
    Y_tt,
    Z_tt,
    mu
):
    upper_rhs = IDX_0 + tt_sub(tt_add(obj_tt, Z_tt), tt_mat(tt_linear_op(tt_adjoint(linear_op_tt), Y_tt), shape=(2, 2)))
    middle_rhs = IDX_1 + tt_sub(bias_tt, tt_mat(tt_linear_op(linear_op_tt, X_tt), shape=(2, 2)))
    newton_rhs = tt_add(upper_rhs, middle_rhs)
    primal_dual_error = tt_inner_prod(newton_rhs, newton_rhs)
    XZ_term = tt_mat_mat_mul(X_tt, Z_tt)
    lower_rhs = IDX_3 + tt_sub(tt_add(XZ_term, tt_transpose(XZ_term)), tt_scale(2 * mu, tt_identity(len(X_tt))))
    newton_rhs = tt_vec(tt_add(newton_rhs, lower_rhs))
    return tt_rank_reduce(tt_scale(-1, newton_rhs), err_bound=0), primal_dual_error


def tt_infeasible_newton_system_lhs(lhs_skeleton, X_tt, Z_tt):
    Z_op_tt = IDX_30 + tt_op_to_mat(tt_add(tt_op_left_from_tt_matrix(Z_tt), tt_op_right_from_tt_matrix(Z_tt)))
    X_op_tt = IDX_33 + tt_op_to_mat(tt_add(tt_op_right_from_tt_matrix(X_tt), tt_op_left_from_tt_matrix(X_tt)))
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
        X_tt,
        Y_tt,
        Z_tt,
        verbose
):
    mu = tt_inner_prod(Z_tt, [0.5 * c for c in X_tt])
    lhs_matrix_tt = tt_infeasible_newton_system_lhs(lhs_skeleton, X_tt, Z_tt)
    rhs_vec_tt, primal_dual_error = tt_infeasible_feas_rhs(
        obj_tt, linear_op_tt, bias_tt, X_tt, Y_tt, Z_tt, 0.5*mu
    )
    Delta_tt, res = tt_amen(lhs_matrix_tt, rhs_vec_tt, verbose=verbose, nswp=3)
    Delta_tt = tt_mat(Delta_tt, shape=(2, 2))
    Delta_X_tt = _tt_get_block(0, 0, Delta_tt)
    Delta_Y_tt = _tt_get_block(0, 1, Delta_tt)
    Delta_Z_tt = _tt_get_block(1, 1, Delta_tt)  # Z should be symmetric but it isn't exactly
    x_step_size, z_step_size = _tt_line_search(X_tt, Z_tt, Delta_X_tt, Delta_Z_tt)
    print(f"Step sizes: {x_step_size} {z_step_size}")

    return (
        tt_rank_reduce(tt_add(X_tt, tt_scale(0.98 * x_step_size, Delta_X_tt)), err_bound=0),
        tt_rank_reduce(tt_add(Y_tt, tt_scale(0.98 * z_step_size, Delta_Y_tt)), err_bound=0),
        tt_rank_reduce(tt_add(Z_tt, tt_scale(0.98 * z_step_size, Delta_Z_tt)), err_bound=0),
        primal_dual_error,
        mu
    )

def _symmetrisation(train_tt: List[np.ndarray]) -> np.ndarray:
    train_tt = tt_rank_reduce(tt_scale(0.5, tt_add(train_tt, tt_transpose(train_tt))), err_bound=0)
    return train_tt


def _tt_line_search(X_tt, Z_tt, Delta_X_tt, Delta_Z_tt, crit=0):
    x_step_size = 1
    z_step_size = 1
    discount = 0.5
    discount_x = False
    discount_z = False
    r = X_tt[0].shape[-1]
    new_X_tt = tt_add(X_tt, Delta_X_tt)
    for iter in range(15):
        val_x, _, _ = tt_min_eig(new_X_tt)
        discount_x = np.greater(val_x, crit)
        if discount_x:
            break
        else:
            new_X_tt[0][:, :, :, r:] *= discount
            x_step_size *= discount
    new_Z_tt = tt_add(Z_tt, Delta_Z_tt)
    for iter in range(15):
        val_z, _, _ = tt_min_eig(new_Z_tt)
        discount_z = np.greater(val_z, crit)
        if discount_z:
            break
        else:
            new_Z_tt[0][:, :, :, r:] *= discount
            z_step_size *= discount
    return discount_x*x_step_size, discount_z*z_step_size


def tt_ipm(
    obj_tt,
    linear_op_tt,
    bias_tt,
    max_iter,
    feasibility_tol=1e-4,
    centrality_tol=1e-3,
    verbose=False
):
    dim = len(obj_tt)
    op_tt = tt_scale(-1, linear_op_tt)
    op_tt_adjoint = IDX_01 + tt_op_to_mat(tt_adjoint(op_tt))
    op_tt = IDX_10 + tt_op_to_mat(op_tt)
    I_mat_tt = tt_op_to_mat(tt_op_right_from_tt_matrix(tt_identity(dim)))
    I_op_tt = IDX_03 + I_mat_tt
    lhs_skeleton = tt_add(op_tt, op_tt_adjoint)
    lhs_skeleton = tt_add(lhs_skeleton, I_op_tt)
    lhs_skeleton = tt_rank_reduce(lhs_skeleton, err_bound=0)
    X_tt = tt_identity(dim)
    Y_tt = tt_zero_matrix(dim)  # [0, Y_1, Y_2, 0]^T
    Z_tt = tt_identity(dim)
    iter = 0
    feasible = False
    for iter in range(max_iter):
        X_tt, Y_tt, Z_tt, pd_error, mu = _tt_ipm_newton_step(
            obj_tt,
            linear_op_tt,
            lhs_skeleton,
            bias_tt,
            X_tt,
            Y_tt,
            Z_tt,
            verbose
        )
        # Symmetry correction
        if verbose:
            print(f"---Step {iter}---")
            print(f"Duality Gap: {100 * np.abs(mu):.4f}%")
            print(f"Primal-Dual error: {pd_error:.8f}")
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
    return X_tt, Y_tt, Z_tt


if __name__ == "__main__":
    np.random.seed(84)
    random_M = tt_random_gaussian([2], shape=(2, 2))
    linear_op_tt = tt_rank_reduce(tt_mask_to_linear_op(tt_add(random_M, tt_transpose(random_M))))
    initial_guess = tt_random_gaussian([2], shape=(2, 2))
    initial_guess = tt_rank_reduce(tt_mat_mat_mul(initial_guess, tt_transpose(initial_guess)))
    bias_tt = tt_rank_reduce(tt_mat(tt_linear_op(linear_op_tt, initial_guess), shape=(2, 2)))
    obj_tt = tt_identity(len(bias_tt))
    _ = tt_ipm(obj_tt, linear_op_tt, bias_tt)
