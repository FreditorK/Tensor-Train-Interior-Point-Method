import sys
import os


from src.tt_eig import tt_min_eig
from src.tt_ineq_check import tt_is_geq

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
    linear_op_tt_adjoint,
    bias_tt,
    linear_op_tt_ineq,
    bias_tt_ineq,
    X_tt,
    Y_tt,
    T_tt,
    Z_tt,
    mu
):
    dual_feas = tt_sub(tt_add(obj_tt, Z_tt), tt_mat(tt_linear_op(linear_op_tt_adjoint, Y_tt), shape=(2, 2)))
    primal_feas_1 = tt_sub(bias_tt, tt_mat(tt_linear_op(linear_op_tt, X_tt), shape=(2, 2)))
    middle_rhs = IDX_1 + primal_feas_1
    if linear_op_tt_ineq is not None and bias_tt_ineq is not None:
        primal_feas_2 = tt_hadamard(T_tt, tt_sub(bias_tt_ineq, tt_sub(linear_op_tt_ineq, X_tt)))
        primal_feas_2 = tt_rank_reduce(tt_sub(primal_feas_2, tt_scale(mu, tt_one_matrix(len(X_tt)))))
        middle_rhs = tt_add(middle_rhs, IDX_2 + primal_feas_2)
    upper_rhs = IDX_0 + dual_feas
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
        lhs_skeleton,
        linear_op_tt,
        linear_op_tt_adjoint,
        bias_tt,
        linear_op_tt_ineq,
        bias_tt_ineq,
        X_tt,
        Y_tt,
        T_tt,
        Z_tt,
        tol,
        verbose
):
    mu = tt_inner_prod(Z_tt, [0.5 * c for c in X_tt])
    lhs_matrix_tt = tt_infeasible_newton_system_lhs(lhs_skeleton, X_tt, Z_tt)
    rhs_vec_tt, primal_dual_error = tt_infeasible_feas_rhs(
        obj_tt, linear_op_tt, linear_op_tt_adjoint, bias_tt, linear_op_tt_ineq, bias_tt_ineq, X_tt, Y_tt, T_tt, Z_tt, 0.5 * mu
    )
    Delta_tt, res = tt_amen(lhs_matrix_tt, rhs_vec_tt, verbose=verbose, nswp=22)
    Delta_tt = tt_mat(Delta_tt, shape=(2, 2))
    Delta_X_tt = tt_rank_reduce(_tt_get_block(0, 0, Delta_tt), err_bound=tol)
    Delta_Y_tt = tt_rank_reduce(_tt_get_block(0, 1, Delta_tt), err_bound=tol)
    Delta_Z_tt = tt_rank_reduce(_tt_get_block(1, 1, Delta_tt), err_bound=tol)
    x_step_size, z_step_size = _tt_line_search(X_tt, Z_tt, Delta_X_tt, Delta_Z_tt, linear_op_tt_ineq, bias_tt_ineq)
    if linear_op_tt_ineq is not None and bias_tt_ineq is not None:
        Delta_T_tt = tt_rank_reduce(_tt_get_block(1, 0, Delta_tt), err_bound=tol)
        T_tt = tt_rank_reduce(tt_add(T_tt, tt_scale(0.98 * z_step_size, Delta_T_tt)), err_bound=tol)

    if verbose:
        print(f"Step sizes: {x_step_size} {z_step_size}")

    # TODO: Mind the error bound here! May lead to inaccuracies and more iterations
    return (
        tt_rank_reduce(tt_add(X_tt, tt_scale(0.98 * x_step_size, Delta_X_tt)), err_bound=tol),
        tt_rank_reduce(tt_add(Y_tt, tt_scale(0.98 * z_step_size, Delta_Y_tt)), err_bound=tol),
        T_tt,
        tt_rank_reduce(tt_add(Z_tt, tt_scale(0.98 * z_step_size, Delta_Z_tt)), err_bound=tol),
        primal_dual_error,
        mu
    )


def _tt_line_search(X_tt, Z_tt, Delta_X_tt, Delta_Z_tt, linear_op_tt_ineq, bias_tt_ineq, crit=1e-18):
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

    if linear_op_tt_ineq is not None and bias_tt_ineq is not None:
        for iter in range(15):
            _, val_x, _ = tt_is_geq(linear_op_tt_ineq, new_X_tt, bias_tt_ineq)
            discount_x = np.greater(val_x, crit)
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
    return discount_x*x_step_size, discount_z*z_step_size


def tt_ipm(
    obj_tt,
    linear_op_tt,
    linear_op_tt_adjoint,
    bias_tt,
    linear_op_tt_ineq=None,
    linear_op_tt_adjoint_ineq=None,
    bias_tt_ineq=None,
    max_iter=100,
    feasibility_tol=1e-4,
    centrality_tol=1e-3,
    verbose=False
):
    dim = len(obj_tt)
    op_tt = tt_scale(-1, linear_op_tt)
    op_tt_adjoint = IDX_01 + tt_op_to_mat(tt_scale(-1, linear_op_tt_adjoint))
    op_tt = IDX_10 + tt_op_to_mat(op_tt)
    I_mat_tt = tt_op_to_mat(tt_op_right_from_tt_matrix(tt_identity(dim)))
    I_op_tt = IDX_03 + I_mat_tt
    lhs_skeleton = tt_add(op_tt, op_tt_adjoint)
    lhs_skeleton = tt_add(lhs_skeleton, I_op_tt)
    if linear_op_tt_ineq is not None and bias_tt_ineq is not None:
        op_tt_ineq = tt_scale(-1, linear_op_tt)
        op_tt_ineq_adjoint = IDX_02 + tt_op_to_mat(tt_scale(-1, linear_op_tt_adjoint_ineq))
        op_tt_ineq = IDX_20 + tt_op_to_mat(op_tt_ineq)
        lhs_skeleton = tt_add(lhs_skeleton, op_tt_ineq)
        lhs_skeleton = tt_add(lhs_skeleton, op_tt_ineq_adjoint)
    lhs_skeleton = tt_rank_reduce(lhs_skeleton, err_bound=0)
    X_tt = tt_identity(dim)
    Y_tt = tt_zero_matrix(dim)
    T_tt = tt_zero_matrix(dim)
    Z_tt = tt_identity(dim)
    iter = 0
    feasible = False
    for iter in range(max_iter):
        X_tt, Y_tt, T_tt, Z_tt, pd_error, mu = _tt_ipm_newton_step(
            obj_tt,
            lhs_skeleton,
            linear_op_tt,
            linear_op_tt_adjoint,
            bias_tt,
            linear_op_tt_ineq,
            bias_tt_ineq,
            X_tt,
            Y_tt,
            T_tt,
            Z_tt,
            feasibility_tol,
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
