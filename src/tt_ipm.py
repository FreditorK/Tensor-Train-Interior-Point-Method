import sys
import os

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


def tt_infeasible_newton_system_rhs(obj_tt, linear_op_tt, bias_tt, X_tt, Y_tt, Z_tt, mu):
    upper_rhs = IDX_0 + tt_sub(tt_add(Z_tt, obj_tt), tt_mat(tt_linear_op(tt_adjoint(linear_op_tt), Y_tt), shape=(2, 2)))
    middle_rhs = IDX_1 + tt_sub(bias_tt, tt_mat(tt_linear_op(linear_op_tt, X_tt), shape=(2, 2)))
    lower_rhs = IDX_3 + tt_sub(tt_mat_mat_mul(Z_tt, X_tt), tt_scale(mu, tt_identity(len(X_tt))))
    newton_rhs = tt_add(upper_rhs, middle_rhs)
    newton_rhs = tt_add(newton_rhs, lower_rhs)
    return tt_rank_reduce(tt_scale(-1, tt_vec(newton_rhs)))


def tt_infeasible_newton_system_lhs(lhs_skeleton, X_tt, Z_tt):
    Z_op_tt = IDX_30 + tt_op_to_mat(tt_op_from_tt_matrix(Z_tt))
    X_op_tt = IDX_33 + tt_op_to_mat(tt_op_from_tt_matrix(X_tt))
    newton_system = tt_add(lhs_skeleton, Z_op_tt)
    newton_system = tt_add(newton_system, X_op_tt)
    return tt_rank_reduce(newton_system)


def _tt_get_block(i, j, block_matrix_tt):
    first_core = block_matrix_tt[0][:, i, j, :]
    first_core = np.einsum("ab, bcde -> acde", first_core, block_matrix_tt[1])
    return [first_core] + block_matrix_tt[2:]


def _tt_ipm_newton_step(obj_tt, linear_op_tt, lhs_skeleton, bias_tt, XZ_tt, Y_tt, verbose, tol=1e-5):
    X_tt = _tt_get_block(0, 0, XZ_tt)
    Z_tt = _tt_get_block(1, 1, XZ_tt)
    mu = np.divide(tt_inner_prod(Z_tt, [0.5 * c for c in X_tt]), 2)
    rhs_vec_tt = tt_infeasible_newton_system_rhs(obj_tt, linear_op_tt, bias_tt, X_tt, Y_tt, Z_tt, mu)
    error = np.divide(tt_inner_prod(rhs_vec_tt, rhs_vec_tt), 3)
    if verbose:
        print(f"KKT error: {error:.2f}")
    if np.less(error, tol):
        return 0, mu, True
    lhs_matrix_tt = tt_infeasible_newton_system_lhs(lhs_skeleton, X_tt, Z_tt)
    Delta_tt, res = tt_amen(lhs_matrix_tt, rhs_vec_tt, verbose=verbose, nswp=15)
    return tt_mat(Delta_tt, shape=(2, 2)), mu, False


def _tt_psd_step(XZ_tt, Delta_XZ_tt, step_size=0.5, discount=0.75, max_num_steps=10, tol=1e-6):
    # Projection to PSD-cone
    for iter in range(max_num_steps):
        new_XZ_tt = tt_rank_reduce(tt_add(XZ_tt, tt_scale(step_size, Delta_XZ_tt)))
        if tt_is_psd(new_XZ_tt, max_num_steps, tol):
            return tt_rank_reduce(tt_add(XZ_tt, tt_scale(0.95 * step_size, Delta_XZ_tt))), step_size
        step_size *= discount
    #print(f"Final step size: {step_size}, Error: {last_err}")
    return XZ_tt, step_size


def _symmetrisation(Delta_XZ_tt):
    Delta_XZ_tt = tt_rank_reduce(tt_scale(0.5, tt_add(Delta_XZ_tt, tt_transpose(Delta_XZ_tt))))
    return Delta_XZ_tt


def _get_xz_block(XYZ_tt):
    new_index_block = np.zeros_like(XYZ_tt[0])
    new_index_block[:, 0, 0] += XYZ_tt[0][:, 0, 0]
    new_index_block[:, 1, 1] += XYZ_tt[0][:, 1, 1]
    return [new_index_block] + XYZ_tt[1:]


def tt_ipm(obj_tt, linear_op_tt, bias_tt, max_iter, beta=8e-4, verbose=False):
    dim = len(obj_tt)
    beta = beta ** (1 / (dim - 1))
    op_tt = tt_scale(-1, linear_op_tt)
    op_tt_adjoint = IDX_01 + tt_op_to_mat(tt_adjoint(op_tt))
    op_tt = IDX_10 + tt_op_to_mat(op_tt)
    I_mat_tt = tt_op_to_mat(tt_op_from_tt_matrix(tt_identity(dim)))
    I_op_tt = IDX_03 + I_mat_tt
    lhs_skeleton = tt_add(op_tt, op_tt_adjoint)
    lhs_skeleton = tt_add(lhs_skeleton, I_op_tt)
    # Tikhononv regularization
    lhs_skeleton = tt_rank_reduce(
        tt_add(lhs_skeleton, [beta * np.eye(4).reshape(1, 4, 4, 1) for _ in range(len(lhs_skeleton))]))
    XZ_tt = tt_identity(dim + 1)  # [X, 0, 0, Z]^T
    Y_tt = tt_zeros(dim, shape=(2, 2))  # [0, Y_1, Y_2, 0]^T
    iter = 0
    for iter in range(max_iter):
        #print(np.round(tt_matrix_to_matrix(XZ_tt), decimals=2))
        Delta_tt, mu, stopping_crit = _tt_ipm_newton_step(obj_tt, linear_op_tt, lhs_skeleton, bias_tt, XZ_tt, Y_tt,
                                                          verbose)
        if verbose:
            print(f"Duality Gap: {abs(mu):.3f}%")
        if stopping_crit:
            break
        Delta_XZ_tt = _get_xz_block(Delta_tt)
        Delta_XZ_tt = _symmetrisation(Delta_XZ_tt)
        XZ_tt, alpha = _tt_psd_step(XZ_tt, Delta_XZ_tt)
        Delta_Y_tt = _tt_get_block(0, 1, Delta_tt)
        Y_tt = tt_rank_reduce(tt_add(Y_tt, tt_scale(0.95 * alpha, Delta_Y_tt)))
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
