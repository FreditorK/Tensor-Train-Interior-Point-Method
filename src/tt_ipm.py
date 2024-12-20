import sys
import os

sys.path.append(os.getcwd() + '/../')

from src.tt_ops import *
from src.tt_ops import tt_rank_reduce
from src.tt_amen import tt_block_amen
from src.tt_eig import tt_min_eig
from src.tt_ineq_check import tt_is_geq, tt_is_geq_


def tt_infeasible_feas_rhs(
    vec_obj_tt,
    mat_lin_op_tt,
    mat_lin_op_tt_adj,
    vec_bias_tt,
    mat_lin_op_tt_ineq,
    mat_lin_op_tt_ineq_adj,
    vec_bias_tt_ineq,
    X_tt,
    vec_Y_tt,
    T_tt,
    Z_tt,
    mu,
    tol,
    feasibility_tol,
    active_ineq
):
    rhs = {}
    idx_add = int(active_ineq)
    vec_X_tt = tt_vec(X_tt)
    dual_feas = tt_sub(tt_matrix_vec_mul(mat_lin_op_tt_adj, vec_Y_tt), tt_add(tt_vec(Z_tt), vec_obj_tt))
    primal_feas = tt_rank_reduce(tt_sub(tt_matrix_vec_mul(mat_lin_op_tt, vec_X_tt), vec_bias_tt), err_bound=tol) # primal feasibility
    primal_error = tt_inner_prod(primal_feas, primal_feas)
    if primal_error > feasibility_tol:
        rhs[1] = primal_feas

    if active_ineq:
        vec_T_tt = tt_vec(T_tt)
        dual_feas = tt_add(dual_feas, tt_matrix_vec_mul(mat_lin_op_tt_ineq_adj, vec_T_tt))
        primal_feas_ineq = tt_hadamard(vec_T_tt, tt_sub(tt_matrix_vec_mul(mat_lin_op_tt_ineq, vec_X_tt), vec_bias_tt_ineq))
        # TODO: Does mu 1 not also be under mat_lin_op_tt_ineq, need to adjust mu 1 to have zeros where L(X) has zeros
        one = tt_matrix_vec_mul(mat_lin_op_tt_ineq_adj, [np.ones((1, 2, 1)).reshape(1, 2, 1) for _ in vec_X_tt])
        primal_feas_ineq = tt_rank_reduce(tt_add(primal_feas_ineq, tt_scale(mu, one)), err_bound=min(tol, 0.5*mu))
        primal_ineq_error  = tt_inner_prod(primal_feas_ineq, primal_feas_ineq)
        if primal_ineq_error > 2*tol:
            rhs[2] = primal_feas_ineq
            primal_error += primal_ineq_error

    dual_feas =  tt_rank_reduce(dual_feas, err_bound=tol)
    dual_error = tt_inner_prod(dual_feas, dual_feas)
    XZ_term = tt_mat_mat_mul(X_tt, Z_tt)
    centrality = tt_vec(tt_add(XZ_term, tt_transpose(XZ_term)))
    centrality = tt_rank_reduce(tt_sub(tt_scale(2 * mu, tt_vec(tt_identity(len(X_tt)))), centrality), err_bound=min(tol, mu))
    if dual_error > feasibility_tol:
        rhs[0] = dual_feas
    rhs[2 + idx_add] = centrality
    return rhs, primal_error + dual_error


def tt_infeasible_newton_system_lhs(
        lhs_skeleton,
        X_tt,
        Z_tt,
        T_tt,
        mat_lin_op_tt_ineq,
        vec_bias_tt_ineq,
        tol,
        active_ineq
):
    idx_add = int(active_ineq)
    identity = tt_identity(len(Z_tt))
    lhs_skeleton[(2 + idx_add, 1)] = tt_rank_reduce(tt_add(tt_kron(identity, Z_tt), tt_kron(tt_transpose(Z_tt), identity)), err_bound=tol)
    lhs_skeleton[(2 + idx_add, 2 + idx_add)] = tt_rank_reduce(tt_add(tt_kron(tt_transpose(X_tt), identity), tt_kron(identity, X_tt)), err_bound=tol)
    if active_ineq:
        ineq_res_tt = tt_sub(vec_bias_tt_ineq, tt_matrix_vec_mul(mat_lin_op_tt_ineq, tt_vec(X_tt)))
        mat_ineq_res_op_tt = tt_diag(ineq_res_tt)
        mat_T_op_tt = tt_diag(tt_vec(T_tt))
        mat_T_comp_linear_op_tt_ineq = tt_mat_mat_mul(mat_T_op_tt, tt_scale(-1, mat_lin_op_tt_ineq))
        lhs_skeleton[(2, 2)] = tt_rank_reduce(mat_ineq_res_op_tt, err_bound=tol)
        lhs_skeleton[(2, 1)] = tt_rank_reduce(mat_T_comp_linear_op_tt_ineq, err_bound=tol)
    return lhs_skeleton


def _tt_symmetrise(matrix_tt, err_bound):
    return tt_rank_reduce(tt_scale(0.5, tt_add(matrix_tt, tt_transpose(matrix_tt))), err_bound=err_bound)


def _tt_get_block(i, block_matrix_tt):
    return  block_matrix_tt[:-1] + [block_matrix_tt[-1][:, i]]

def _tt_ipm_newton_step(
        vec_obj_tt,
        lhs_skeleton,
        mat_lin_op_tt,
        mat_lin_op_tt_adj,
        vec_bias_tt,
        mat_lin_op_tt_ineq,
        mat_lin_op_tt_ineq_adj,
        vec_bias_tt_ineq,
        X_tt,
        vec_Y_tt,
        T_tt,
        Z_tt,
        tol,
        feasibility_tol,
        active_ineq,
        verbose,
        eps = 1e-10,
):
    mu = tt_inner_prod(Z_tt, [0.5 * c for c in X_tt])
    idx_add = int(active_ineq)
    lhs_matrix_tt = tt_infeasible_newton_system_lhs(
        lhs_skeleton,
        X_tt,
        Z_tt,
        T_tt,
        mat_lin_op_tt_ineq,
        vec_bias_tt_ineq,
        tol,
        active_ineq
    )
    rhs_vec_tt, primal_dual_error = tt_infeasible_feas_rhs(
        vec_obj_tt,
        mat_lin_op_tt,
        mat_lin_op_tt_adj,
        vec_bias_tt,
        mat_lin_op_tt_ineq,
        mat_lin_op_tt_ineq_adj,
        vec_bias_tt_ineq,
        X_tt,
        vec_Y_tt,
        T_tt,
        Z_tt,
        0.5 * mu,
        tol,
        feasibility_tol,
        active_ineq
    )
    Delta_tt, res = tt_block_amen(lhs_matrix_tt, rhs_vec_tt, kickrank=2, eps=eps, verbose=verbose)
    vec_Delta_Y_tt = tt_rank_reduce(_tt_get_block(0, Delta_tt), err_bound=tol)
    Delta_T_tt = tt_rank_reduce(tt_mat(_tt_get_block(2, Delta_tt)), err_bound=tol) if active_ineq else None
    Delta_X_tt = tt_rank_reduce(tt_mat(_tt_get_block(1, Delta_tt)), err_bound=tol)
    Delta_Z_tt = tt_rank_reduce(tt_mat(_tt_get_block(2+idx_add, Delta_tt)), err_bound=tol)
    if np.greater(res, eps):
        Delta_X_tt = _tt_symmetrise(Delta_X_tt, tol)
        Delta_Z_tt = _tt_symmetrise(Delta_Z_tt, tol)
    x_step_size, z_step_size = _tt_line_search(X_tt, T_tt, Z_tt, Delta_X_tt, Delta_T_tt, Delta_Z_tt, mat_lin_op_tt_ineq, vec_bias_tt_ineq, active_ineq)
    X_tt = tt_rank_reduce(tt_add(X_tt, tt_scale(0.98 * x_step_size, Delta_X_tt)), err_bound=0.5*tol)
    vec_Y_tt = tt_rank_reduce(tt_add(vec_Y_tt, tt_scale(0.98 * z_step_size, vec_Delta_Y_tt)), err_bound=tol)
    Z_tt = tt_rank_reduce(tt_add(Z_tt, tt_scale(0.98 * z_step_size, Delta_Z_tt)), err_bound=0.5*tol)
    if active_ineq:
        # FIXME: Note that T_tt should grow large on the zeros of b - L_ineq(X_tt)
        T_tt = tt_rank_reduce(tt_add(T_tt, tt_scale(0.98 * z_step_size, Delta_T_tt)), err_bound=tol)

    if verbose:
        print(f"Step sizes: {x_step_size}, {z_step_size}")

    print("Report ---")
    print("Y")
    print(np.round(tt_matrix_to_matrix(tt_mat(vec_Y_tt)), decimals=3))
    if active_ineq:
        print("T")
        print(np.round(tt_matrix_to_matrix(T_tt), decimals=3))
    print("X")
    print(np.round(tt_matrix_to_matrix(X_tt), decimals=3))
    print("Z")
    print(np.round(tt_matrix_to_matrix(Z_tt), decimals=3))

    return X_tt, vec_Y_tt, T_tt, Z_tt, primal_dual_error, mu


def _tt_line_search(
        X_tt,
        T_tt,
        Z_tt,
        Delta_X_tt,
        Delta_T_tt,
        Delta_Z_tt,
        lin_op_tt_ineq,
        vec_bias_tt_ineq,
        active_ineq,
        iters=15,
        crit=1e-10
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
        # FIXME: The eigenvalue could give an upper bound on error bound to round with
        # FIXME: Can we use tt_eig somehow to round while maintaining psdness?
        discount_x = np.greater(val_x, crit)
        if discount_x:
            break
        else:
            new_X_tt[0][:, :, :, r:] *= discount
            x_step_size *= discount

    if active_ineq and discount_x:
        for iter in range(iters):
            discount_x, val, _ = tt_is_geq(lin_op_tt_ineq, new_X_tt, vec_bias_tt_ineq, crit=crit)
            if discount_x:
                break
            else:
                new_X_tt[0][:, :, :, r:] *= discount
                x_step_size *= discount

    r = Z_tt[0].shape[-1]
    new_Z_tt = tt_add(Z_tt, Delta_Z_tt)
    for iter in range(iters):
        val_z, _, _ = tt_min_eig(new_Z_tt)
        discount_z = np.greater(val_z, crit)
        if discount_z:
            break
        else:
            new_Z_tt[0][:, :, :, r:] *= discount
            z_step_size *= discount

    if active_ineq and discount_z:
        r = T_tt[0].shape[-1]
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
    feasibility_tol=1e-5,
    centrality_tol=1e-3,
    verbose=False
):
    dim = len(obj_tt)
    feasibility_tol = feasibility_tol / np.sqrt(dim)
    centrality_tol = centrality_tol / np.sqrt(dim)
    op_tol = 0.5*min(feasibility_tol, centrality_tol)
    active_ineq = lin_op_tt_ineq is not None or lin_op_tt_ineq_adj is not None or bias_tt_ineq is not None
    obj_tt = tt_rank_reduce(tt_vec(obj_tt), err_bound=op_tol)
    bias_tt = tt_rank_reduce(tt_vec(bias_tt), err_bound=op_tol)
    lhs_skeleton = {}
    lhs_skeleton[(0, 0)] = tt_rank_reduce(tt_scale(-1, lin_op_tt_adj), err_bound=op_tol)
    lhs_skeleton[(1, 1)] = tt_rank_reduce(tt_scale(-1, lin_op_tt), err_bound=op_tol)
    lhs_skeleton[(0, 2 + int(active_ineq))] = tt_identity(2*dim)
    if active_ineq:
        lhs_skeleton[(0, 2)] = tt_rank_reduce(tt_scale(-1, lin_op_tt_ineq_adj), err_bound=op_tol)
        bias_tt_ineq = tt_rank_reduce(tt_vec(bias_tt_ineq), err_bound=op_tol)
    X_tt = tt_identity(dim)
    vec_Y_tt = [np.zeros((1, 2, 1)) for _ in range(2*dim)]
    T_tt = tt_one_matrix(dim)
    if active_ineq:
        T_tt = tt_mat(tt_matrix_vec_mul(lin_op_tt_ineq_adj, tt_vec(T_tt)))
    Z_tt = tt_identity(dim)
    iter = 0
    for iter in range(1, max_iter):
        X_tt, vec_Y_tt, T_tt, Z_tt, pd_error, mu = _tt_ipm_newton_step(
            obj_tt,
            lhs_skeleton,
            lin_op_tt,
            lin_op_tt_adj,
            bias_tt,
            lin_op_tt_ineq,
            lin_op_tt_ineq_adj,
            bias_tt_ineq,
            X_tt,
            vec_Y_tt,
            T_tt,
            Z_tt,
            op_tol,
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
                f"      vec(Y_tt): {tt_ranks(vec_Y_tt)}, T_tt: {tt_ranks(T_tt)} \n"
            )

        if np.less(pd_error, feasibility_tol) and np.less(np.abs(mu), centrality_tol):
                break
    if verbose:
        print(f"---Terminated---")
        print(f"Converged in {iter} iterations.")
    return X_tt, tt_mat(vec_Y_tt), T_tt, Z_tt
