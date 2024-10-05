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
        newton_rhs = IDX_3 + tt_sub(tt_mat_mat_mul(X_tt, Z_tt), tt_scale(mu, tt_identity(len(X_tt))))
        primal_dual_error = 0
    else:
        upper_rhs = IDX_0 + tt_sub(tt_add(obj_tt, Z_tt),
                                   tt_mat(tt_linear_op(tt_adjoint(linear_op_tt), Y_tt), shape=(2, 2)))
        middle_rhs = IDX_1 + tt_sub(bias_tt, tt_mat(tt_linear_op(linear_op_tt, X_tt), shape=(2, 2)))
        XZ_term = tt_mat_mat_mul(X_tt, Z_tt)
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
    Delta_tt, res = tt_amen(lhs_matrix_tt, rhs_vec_tt, verbose=verbose, nswp=22)
    return tt_mat(Delta_tt, shape=(2, 2)), primal_dual_error, mu


def tt_psd_step(matrix_tt: List[np.array], Delta_tt, block_size=2, num_iter=500, mu=2, tol=1e-5):
    # FIXME: Could potentially change to mu=1

    block_matrix_tt = [np.ones((1, block_size, 2, 1))] + matrix_tt
    discount_block = np.array([0.5 ** i for i in range(2 * block_size)]).reshape(1, block_size, 2, 1)
    block_Delta_tt = [discount_block] + Delta_tt
    block_matrix_tt = tt_add(block_matrix_tt, block_Delta_tt)
    _, block_eig_vals = tt_randomised_block_min_eigentensor(block_matrix_tt, num_iter=num_iter, mu=mu, tol=tol)
    min_index = np.argmin(np.abs(block_eig_vals - (block_eig_vals < tol)))
    multi_min_index = np.unravel_index(min_index, block_eig_vals.shape)
    if np.greater(block_eig_vals[multi_min_index], 0):
        factor = block_matrix_tt[0][:, multi_min_index[1], multi_min_index[2], :]
        matrix_tt = [np.einsum("ab, bcde -> acde", factor, block_matrix_tt[1])] + block_matrix_tt[2:]
        return matrix_tt, discount_block[multi_min_index]

    return matrix_tt, (0.5) ** (2 * block_size + 2)


def _tt_psd_step(
    X_tt,
    Y_tt,
    Z_tt,
    Delta_X_tt,
    Delta_Y_tt,
    Delta_Z_tt,
    block_size=5,
    tol=1e-5
):
    _, step_size_x = tt_psd_step(X_tt, Delta_X_tt, block_size=block_size,
                                        num_iter=1500, tol=tol)
    Delta_X_tt[0] *= 0.95*step_size_x
    new_X_tt = tt_add(X_tt, Delta_X_tt)
    _, step_size_z = tt_psd_step(Z_tt, Delta_Z_tt, block_size=block_size,
                                        num_iter=1500, tol=tol)
    Delta_Z_tt[0] *= 0.95*step_size_z
    new_Z_tt = tt_add(Z_tt, Delta_Z_tt)
    new_Y_tt = tt_add(Y_tt, tt_scale(0.95*step_size_z, Delta_Y_tt))
    print(f"Step Size: {step_size_x}, {step_size_z}")
    return (
        tt_rank_reduce(new_X_tt, err_bound=0),
        tt_rank_reduce(new_Y_tt, err_bound=0),
        tt_rank_reduce(new_Z_tt, err_bound=0)
    )


"""
def _tt_psd_step(
    X_tt,
    Y_tt,
    Z_tt,
    Delta_X_tt,
    Delta_Y_tt,
    Delta_Z_tt,
    block_size=5,
    tol=1e-5
):
    x_step_size = 1
    z_step_size = 1
    discount = 0.5
    discount_x = True
    discount_z = True
    for iter in range(10):
        new_X_tt = tt_add(X_tt, Delta_X_tt)
        new_Z_tt = tt_add(Z_tt, Delta_Z_tt)
        discount_x = np.all(np.linalg.eigvals(tt_matrix_to_matrix(new_X_tt)) >= 0)
        discount_z = np.all(np.linalg.eigvals(tt_matrix_to_matrix(new_Z_tt)) >= 0)
        if ~discount_x:
            x_step_size *= discount
            Delta_X_tt[0] *= discount
        if ~discount_z:
            z_step_size *= discount
            Delta_Z_tt[0] *= discount
        if discount_x and discount_z:
            Delta_X_tt[0] *= 0.95
            Delta_Z_tt[0] *= 0.95
            print(f"Step Size: {x_step_size}, {z_step_size}")
            return (
                tt_rank_reduce(tt_add(X_tt, Delta_X_tt), err_bound=0),
                tt_rank_reduce(tt_add(Y_tt, tt_scale(z_step_size, Delta_Y_tt)), err_bound=0),
                tt_rank_reduce(tt_add(Z_tt, Delta_Z_tt), err_bound=0)
            )
    Delta_X_tt[0] *= discount_x
    Delta_Z_tt[0] *= discount_z
    print(f"Step Size: {x_step_size}, {z_step_size}")
    return (
        tt_rank_reduce(tt_add(X_tt, Delta_X_tt), err_bound=0),
        tt_rank_reduce(tt_add(Y_tt, tt_scale(z_step_size, Delta_Y_tt)), err_bound=0),
        tt_rank_reduce(tt_add(Z_tt, Delta_Z_tt), err_bound=0)
    )
"""


def _symmetrisation(train_tt):
    train_tt = tt_rank_reduce(tt_scale(0.5, tt_add(train_tt, tt_transpose(train_tt))), err_bound=0)
    return train_tt


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
        tt_add(lhs_skeleton, [tikhonov_param * np.eye(4).reshape(1, 4, 4, 1) for _ in range(len(lhs_skeleton))]),
        err_bound=0)
    X_tt = tt_identity(dim)
    Y_tt = tt_zeros(dim, shape=(2, 2))  # [0, Y_1, Y_2, 0]^T
    Z_tt = tt_identity(dim)
    iter = 0
    centering_param = 0.5
    Delta_X_tt = tt_zeros(dim, shape=(2, 2))
    Delta_Z_tt = tt_zeros(dim, shape=(2, 2))
    feasible = False
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
        condition = min(1, (pd_error / mu) ** 3)
        centering_param = interpol_damp * centering_param + (1 - interpol_damp) * condition
        Delta_X_tt = _symmetrisation(_tt_get_block(0, 0, Delta_tt))
        Delta_Y_tt = _tt_get_block(0, 1, Delta_tt)
        Delta_Z_tt = _symmetrisation(_tt_get_block(1, 1, Delta_tt))
        if verbose:
            print(f"---Step {iter}---")
            print("Centering Param: ", centering_param)
            print(f"Duality Gap: {100 * abs(mu):.4f}%")
            print(f"Primal-Dual error: {pd_error:.4f}")
            #print("X_tt:")
            #print(np.round(tt_matrix_to_matrix(X_tt), decimals=2))
            #print("Delta_X_tt:")
            #print(np.round(tt_matrix_to_matrix(Delta_X_tt), decimals=2))
            #print("Z_tt:")
            #print(np.round(tt_matrix_to_matrix(Z_tt), decimals=2))
            #print("Delta_Z_tt:")
            #print(np.round(tt_matrix_to_matrix(Delta_Z_tt), decimals=2))
        X_tt, Y_tt, Z_tt = _tt_psd_step(X_tt, Y_tt, Z_tt, Delta_X_tt, Delta_Y_tt, Delta_Z_tt)
        if np.less(pd_error, feasibility_tol):
            feasible = True
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
