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


def tt_infeasible_newton_system_rhs(
    obj_tt,
    linear_op_tt,
    bias_tt,
    X_tt,
    Y_tt,
    Z_tt,
    Delta_X_tt,
    Delta_Z_tt,
    mu,
    feasible
):
    # Mehrotra's Aggregated System
    if feasible:
        # If primal-dual error is beneath tolerance, quicken up newton_system construction by assuming zero pd-error
        XZ_term = tt_mat_mat_mul(X_tt, Z_tt)
        XZ_term = tt_add(XZ_term, tt_mat_mat_mul(Delta_X_tt, Delta_Z_tt))
        newton_rhs = IDX_3 + tt_sub(XZ_term, tt_scale(mu, tt_identity(len(X_tt))))
        primal_dual_error = 0
    else:
        upper_rhs = IDX_0 + tt_sub(tt_add(obj_tt, Z_tt),
                                   tt_mat(tt_linear_op(tt_adjoint(linear_op_tt), Y_tt), shape=(2, 2)))
        middle_rhs = IDX_1 + tt_sub(bias_tt, tt_mat(tt_linear_op(linear_op_tt, X_tt), shape=(2, 2)))
        newton_rhs = tt_add(upper_rhs, middle_rhs)
        primal_dual_error = tt_inner_prod(newton_rhs, newton_rhs)
        XZ_term = tt_mat_mat_mul(X_tt, Z_tt)
        XZ_term = tt_add(XZ_term, tt_mat_mat_mul(Delta_X_tt, Delta_Z_tt))
        lower_rhs = IDX_3 + tt_sub(XZ_term, tt_scale(mu, tt_identity(len(X_tt))))
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
    X_tt,
    Y_tt,
    Z_tt,
    Delta_X_tt,
    Delta_Z_tt,
    centering_param,
    feasible,
    verbose
):
    mu = tt_inner_prod(Z_tt, [0.5 * c for c in X_tt])
    rhs_vec_tt, primal_dual_error = tt_infeasible_newton_system_rhs(
        obj_tt, linear_op_tt, bias_tt, X_tt, Y_tt, Z_tt, Delta_X_tt, Delta_Z_tt, centering_param * mu, feasible
    )
    lhs_matrix_tt = tt_infeasible_newton_system_lhs(lhs_skeleton, X_tt, Z_tt)
    t0 = time.time()
    Delta_tt, res = tt_amen(lhs_matrix_tt, rhs_vec_tt, verbose=verbose, nswp=22)
    print(f"AMeN-time: {time.time() - t0:2f}s")
    return tt_mat(Delta_tt, shape=(2, 2)), primal_dual_error, mu


def _tt_psd_step(
    X_tt,
    Y_tt,
    Z_tt,
    Delta_X_tt,
    Delta_Y_tt,
    Delta_Z_tt
):
    x_step_size = 1
    z_step_size = 1
    discount = 0.5
    discount_x = False
    discount_z = False
    for iter in range(10):
        if ~discount_x:
            new_X_tt = tt_add(X_tt, Delta_X_tt)
            val_x, _, _ = tt_min_eig(new_X_tt)
            discount_x = np.greater(val_x, 0)
            x_step_size *= discount
            Delta_X_tt[0] *= discount
        if ~discount_z:
            new_Z_tt = tt_add(Z_tt, Delta_Z_tt)
            val_z, _, _ = tt_min_eig(new_Z_tt)
            discount_z = np.greater(val_z, 0)
            z_step_size *= discount
            Delta_Z_tt[0] *= discount
        if discount_x and discount_z:
            Delta_X_tt[0] *= 0.95
            Delta_Z_tt[0] *= 0.95
            print(f"Step Size: {x_step_size}, {z_step_size}")
            return (
                tt_rank_reduce(tt_add(X_tt, Delta_X_tt), err_bound=0),
                tt_rank_reduce(tt_add(Y_tt, tt_scale(max(x_step_size, z_step_size), Delta_Y_tt)), err_bound=0),
                tt_rank_reduce(tt_add(Z_tt, Delta_Z_tt), err_bound=0)
            )
    Delta_X_tt[0] *= discount_x
    Delta_Z_tt[0] *= discount_z
    print(f"Step Size: {x_step_size}, {z_step_size}")
    return (
        tt_rank_reduce(tt_add(X_tt, Delta_X_tt), err_bound=0),
        tt_rank_reduce(tt_add(Y_tt, tt_scale(max(x_step_size, z_step_size), Delta_Y_tt)), err_bound=0),
        tt_rank_reduce(tt_add(Z_tt, Delta_Z_tt), err_bound=0)
    )


def _symmetrisation(train_tt: List[np.ndarray]) -> np.ndarray:
    train_tt = tt_rank_reduce(tt_scale(0.5, tt_add(train_tt, tt_transpose(train_tt))), err_bound=0)
    return train_tt


def tt_ipm(
    obj_tt,
    linear_op_tt,
    bias_tt,
    max_iter,
    tikhonov_param=1e-4,
    feasibility_tol=1e-4,
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
        tt_add(lhs_skeleton, [tikhonov_param * np.eye(4).reshape(1, 4, 4, 1) for _ in range(len(lhs_skeleton))]),
        err_bound=0
    )
    X_tt = tt_identity(dim)
    Y_tt = tt_zero_matrix(dim)  # [0, Y_1, Y_2, 0]^T
    Z_tt = tt_identity(dim)
    iter = 0
    centering_param = 0.99
    Delta_X_tt = tt_zero_matrix(dim)
    Delta_Z_tt = tt_zero_matrix(dim)
    feasible = False
    prev_mu = 1
    for iter in range(max_iter):
        Delta_tt, pd_error, mu = _tt_ipm_newton_step(
            obj_tt,
            linear_op_tt,
            lhs_skeleton,
            bias_tt,
            X_tt,
            Y_tt,
            Z_tt,
            Delta_X_tt,
            Delta_Z_tt,
            centering_param,
            feasible,
            verbose
        )
        Delta_X_tt = _symmetrisation(_tt_get_block(0, 0, Delta_tt))
        Delta_Y_tt = _tt_get_block(0, 1, Delta_tt)
        # FIXME: Pay close attention here if not symmetric because of numerical errors it is not good
        Delta_Z_tt = _tt_get_block(1, 1, Delta_tt) # Z should be symmetric but it isn't exactly
        if verbose:
            print(f"---Step {iter}---")
            print("Centering Param: ", centering_param)
            print(f"Duality Gap: {100 * abs(mu):.4f}%")
            print(f"Primal-Dual error: {pd_error:.8f}")
        X_tt, Y_tt, Z_tt = _tt_psd_step(X_tt, Y_tt, Z_tt, Delta_X_tt, Delta_Y_tt, Delta_Z_tt)
        pd_error = pd_error
        if np.less(pd_error, feasibility_tol):
            feasible = True
            centering_param = min((0.95*mu/prev_mu)**3, 1)
            prev_mu = mu
            if np.less(mu, centrality_tol):
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
