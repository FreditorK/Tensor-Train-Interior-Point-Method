import copy
import sys
import os

import numpy as np
import scipy.optimize
import scipy.sparse.linalg

sys.path.append(os.getcwd() + '/../')

from src.tt_ops import *
from src.tt_ops import tt_rank_reduce
from src.tt_amen import tt_block_amen,  svd_solve_local_system
from src.tt_eig import tt_min_eig
from src.tt_ineq_check import tt_is_geq, tt_is_geq_, tt_is_psd


def forward_backward_sub(L, b):
    y = scip.linalg.solve_triangular(L, b, lower=True, check_finite=False)
    x = scip.linalg.solve_triangular(L.T, y, lower=False, check_finite=False, overwrite_b=True)
    return x

def ipm_solve_local_system(prev_sol, lhs, rhs, local_auxs, num_blocks, eps):
    k =  num_blocks - 1
    block_dim = lhs.shape[0] // num_blocks
    prev_yt = prev_sol[block_dim:k*block_dim]

    L_eq = -lhs[block_dim:2*block_dim, :block_dim]
    L_Z = lhs[k*block_dim:, :block_dim]
    #L_L_Z = scip.linalg.cholesky(L_Z, check_finite=False, overwrite_a=True, lower=True)
    L_inv = np.linalg.inv(L_Z)
    L_eq_adj = -lhs[:block_dim, block_dim:2*block_dim]
    #I = lhs[:block_dim, k*block_dim:]
    inv_I = np.diag(np.divide(1, np.diagonal(lhs[:block_dim, k*block_dim:])))
    L_X = lhs[k * block_dim:, k * block_dim:]
    #L_L_X = scip.linalg.cholesky(L_X, check_finite=False, overwrite_a=True, lower=True)
    R_d = -rhs[:block_dim]
    R_p = -rhs[block_dim:2*block_dim]
    R_c = -rhs[k * block_dim:]
    K_temp = L_inv  @ L_X
    K = K_temp @ inv_I

    #print("hi", np.linalg.norm(L_Z @ K_temp - L_X))
    k = L_inv @ R_c
    #print("hi 2", np.linalg.norm(L_Z @ k - R_c))
    KR_dmk = K @ R_d - k

    if num_blocks > 3:
        TL_ineq = -lhs[2 * block_dim:3 * block_dim, :block_dim]
        L_ineq_adj = -lhs[:block_dim, 2 * block_dim:3 * block_dim]
        R_ineq = lhs[2 * block_dim:3 * block_dim, 2 * block_dim:3 * block_dim]
        R_t = -rhs[2 * block_dim:3 * block_dim]
        A = np.block([
            [L_eq @ K @ L_eq_adj, L_eq @ K @ L_ineq_adj],
            [TL_ineq @ K @ L_eq_adj, R_ineq + TL_ineq @ K @ L_ineq_adj],
            [local_auxs["y"], np.zeros((block_dim, block_dim))],
            [np.zeros((block_dim, block_dim)), local_auxs["t"]]
        ])
        b = np.block([
            [L_eq @ KR_dmk - R_p],
            [TL_ineq @ KR_dmk - R_t],
            [np.zeros((2*block_dim, 1))]
        ])
        P, _ = np.linalg.qr(np.random.randn(4*block_dim, 2*block_dim))
        rhs = P.T @ (b - A @ prev_yt)
        lhs = P.T @ A
        sol = np.linalg.solve(lhs, rhs)
        yt = sol + prev_yt
        y = yt[:block_dim]
        t = yt[block_dim:]
        R_dmL_eq_adj_yt = R_d - L_eq_adj @ y - L_ineq_adj @ t
        x = K @ R_dmL_eq_adj_yt - k
        z = -inv_I @ R_dmL_eq_adj_yt
        #print()
        #print("---")
        #print(np.linalg.norm(-L_eq @ x + R_p)) # bad
        #print(np.linalg.norm(-L_eq_adj @ y - L_ineq_adj @ t + I @ z + R_d))
        #print(np.linalg.norm(-TL_ineq @ x + R_ineq @ t + R_t)) # bad
        #print(np.linalg.norm(L_Z @ x + L_X @ z + R_c))
        #print("---")
        return np.vstack((x, y, t, z))
    A = np.block([
        [L_eq @ K @ L_eq_adj],
        [local_auxs["y"]]
    ])
    b = np.block([[L_eq @ KR_dmk - R_p],
                    [np.zeros((block_dim, 1))]])
    P, _ = np.linalg.qr(np.random.randn(2 * block_dim, block_dim))
    rhs = P.T @ (b - A @ prev_yt)
    lhs = P.T @ A
    print("Cond: ", np.linalg.cond(lhs))
    sol = np.linalg.solve(lhs, rhs)
    y = sol + prev_yt
    R_dmL_eq_adj_y = R_d - L_eq_adj @ y
    x = K @ R_dmL_eq_adj_y - k
    z = -inv_I @ R_dmL_eq_adj_y

    #print("---")
    #print(np.linalg.norm(- L_eq_adj @ y + I @ z + R_d))
    #print(np.linalg.norm(- L_eq @ x + R_p))
    #print(np.linalg.norm(L_Z @ x + L_X @ z + R_c))
    #print("---")
    return np.vstack((x, y, z))


def tt_preconditioner(Z_tt):
    dim = len(Z_tt)
    rank_one_Z_tt = tt_rank_retraction(tt_diag(tt_diagonal(copy.deepcopy(Z_tt))), [1]*(dim-1))
    root_Z_tt = []
    root_Z_tt_inv = []
    sign = 1
    for i, c in enumerate(rank_one_Z_tt):
        c = sign*np.squeeze(c)
        l, Q = np.linalg.eigh(c)
        if np.all(l < 0) and i < dim-1:
            l *= - 1
            sign *= -1
        else:
            r = np.sum(l > 0)
            l =  l[-r:]
            Q = Q[:, -r:]
        root_Z_tt.append((Q @ np.diag(np.sqrt(l)).reshape(1, 2, 2, 1)))
        root_Z_tt_inv.append((Q @ np.diag(np.divide(1, np.sqrt(l)))).reshape(1, 2, 2, 1))
    print(tt_matrix_to_matrix(root_Z_tt) @ tt_matrix_to_matrix(root_Z_tt).T)
    return root_Z_tt, root_Z_tt_inv

def tt_infeasible_newton_system(
        lhs_skeleton,
        vec_obj_tt,
        X_tt,
        vec_Y_tt,
        Z_tt,
        T_tt,
        mat_lin_op_tt,
        mat_lin_op_tt_adj,
        vec_bias_tt,
        mat_lin_op_tt_ineq,
        mat_lin_op_tt_ineq_adj,
        vec_bias_tt_ineq,
        mu,
        tol,
        feasibility_tol,
        eps,
        active_ineq
):
    idx_add = int(active_ineq)
    scaling_matrix, scaling_matrix_inv = tt_preconditioner(Z_tt) #tt_identity(len(Z_tt))
    L_Z = tt_rank_reduce(tt_add(tt_kron(tt_transpose(scaling_matrix_inv), tt_mat_mat_mul(scaling_matrix, Z_tt)), tt_kron(tt_mat_mat_mul(scaling_matrix, tt_transpose(Z_tt)), tt_transpose(scaling_matrix_inv))), eps=tol)
    L_X = tt_rank_reduce(tt_add(tt_kron(tt_transpose(scaling_matrix_inv), tt_mat_mat_mul(scaling_matrix, X_tt)), tt_kron(tt_mat_mat_mul(scaling_matrix, tt_transpose(X_tt)), tt_transpose(scaling_matrix_inv))), eps=tol)

    if active_ineq:
        ineq_res_tt = tt_sub(vec_bias_tt_ineq, tt_fast_matrix_vec_mul(mat_lin_op_tt_ineq, tt_vec(X_tt), eps))
        mat_ineq_res_op_tt = tt_diag(ineq_res_tt)
        mat_T_op_tt = tt_diag(tt_vec(T_tt))
        lhs_skeleton[(2, 2)] = tt_rank_reduce(mat_ineq_res_op_tt, tol)
        lhs_skeleton[(2, 0)] = tt_fast_mat_mat_mul(mat_T_op_tt, tt_scale(-1, mat_lin_op_tt_ineq), tol)
    lhs_skeleton[(2 + idx_add, 0)] = L_Z
    lhs_skeleton[(2 + idx_add, 2 + idx_add)] = L_X

    rhs = {}
    vec_X_tt = tt_vec(X_tt)
    dual_feas = tt_sub(tt_fast_matrix_vec_mul(mat_lin_op_tt_adj, vec_Y_tt, eps), tt_add(tt_vec(Z_tt), vec_obj_tt))
    primal_feas = tt_rank_reduce(tt_sub(tt_fast_matrix_vec_mul(mat_lin_op_tt, vec_X_tt, eps), vec_bias_tt), tol)  # primal feasibility
    primal_error = tt_inner_prod(primal_feas, primal_feas)
    if primal_error > feasibility_tol:
        rhs[1] = primal_feas

    if active_ineq:
        vec_T_tt = tt_vec(T_tt)
        dual_feas = tt_add(dual_feas, tt_fast_matrix_vec_mul(mat_lin_op_tt_ineq_adj, vec_T_tt, eps))
        primal_feas_ineq = tt_fast_hadammard(vec_T_tt, tt_sub(tt_fast_matrix_vec_mul(mat_lin_op_tt_ineq, vec_X_tt, eps), vec_bias_tt_ineq), tol)
        # TODO: Does mu 1 not also be under mat_lin_op_tt_ineq, need to adjust mu 1 to have zeros where L(X) has zeros
        one = tt_matrix_vec_mul(mat_lin_op_tt_ineq_adj, [np.ones((1, 2, 1)).reshape(1, 2, 1) for _ in vec_X_tt])
        primal_feas_ineq = tt_rank_reduce(tt_add(primal_feas_ineq, tt_scale(mu, one)), tol)
        primal_ineq_error = tt_inner_prod(primal_feas_ineq, primal_feas_ineq)
        if primal_ineq_error > tol:
            rhs[2] = primal_feas_ineq
            primal_error += primal_ineq_error

    dual_feas = tt_rank_reduce(dual_feas, tol)
    dual_error = tt_inner_prod(dual_feas, dual_feas)
    rhs[2 + idx_add] = tt_sub(tt_scale(2 * mu, tt_vec(tt_identity(len(X_tt)))), tt_fast_matrix_vec_mul(L_Z, vec_X_tt, eps))
    if dual_error > feasibility_tol:
        rhs[0] = dual_feas

    return lhs_skeleton, rhs, primal_error + dual_error


def _tt_symmetrise(matrix_tt, err_bound):
    return tt_rank_reduce(tt_scale(0.5, tt_add(matrix_tt, tt_transpose(matrix_tt))), eps=err_bound)


def _tt_get_block(i, block_matrix_tt):
    return  block_matrix_tt[:-1] + [block_matrix_tt[-1][:, i]]

def _tt_ipm_newton_step(
        lag_maps,
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
        local_solver,
        verbose,
        eps
):
    mu = tt_inner_prod(Z_tt, [0.5 * c for c in X_tt])
    op_eps = (1e-4)*eps
    lhs_matrix_tt, rhs_vec_tt, primal_dual_error = tt_infeasible_newton_system(
        lhs_skeleton,
        vec_obj_tt,
        X_tt,
        vec_Y_tt,
        Z_tt,
        T_tt,
        mat_lin_op_tt,
        mat_lin_op_tt_adj,
        vec_bias_tt,
        mat_lin_op_tt_ineq,
        mat_lin_op_tt_ineq_adj,
        vec_bias_tt_ineq,
        0.5 * mu,
        tol,
        feasibility_tol,
        op_eps,
        active_ineq
    )
    idx_add = int(active_ineq)
    Delta_tt, res = tt_block_amen(lhs_matrix_tt, rhs_vec_tt, aux_matrix_blocks=lag_maps, kickrank=2, eps=eps, local_solver=local_solver, verbose=verbose)
    vec_Delta_Y_tt = tt_rank_reduce(_tt_get_block(1, Delta_tt), eps=tol)
    Delta_T_tt = tt_rank_reduce(tt_mat(_tt_get_block(2, Delta_tt)), eps=tol) if active_ineq else None
    Delta_X_tt = tt_rank_reduce(tt_mat(_tt_get_block(0, Delta_tt)), eps=tol)
    Delta_Z_tt = tt_rank_reduce(tt_mat(_tt_get_block(2 + idx_add, Delta_tt)), eps=tol)
    if np.greater(res, eps):
        Delta_X_tt = _tt_symmetrise(Delta_X_tt, tol)
        Delta_Z_tt = _tt_symmetrise(Delta_Z_tt, tol)
    x_step_size, z_step_size = _tt_line_search(X_tt, T_tt, Z_tt, Delta_X_tt, Delta_T_tt, Delta_Z_tt, mat_lin_op_tt_ineq, vec_bias_tt_ineq, active_ineq)
    X_tt = tt_rank_reduce(tt_add(X_tt, tt_scale(0.98 * x_step_size, Delta_X_tt)), eps=0.5 * tol)
    vec_Y_tt = tt_rank_reduce(tt_add(vec_Y_tt, tt_scale(0.98 * z_step_size, vec_Delta_Y_tt)), eps=tol)
    Z_tt = tt_rank_reduce(tt_add(Z_tt, tt_scale(0.98 * z_step_size, Delta_Z_tt)), eps=0.5 * tol)
    if active_ineq:
        # FIXME: Note that T_tt should grow large on the zeros of b - L_ineq(X_tt)
        T_tt = tt_rank_reduce(tt_add(T_tt, tt_scale(0.98 * z_step_size, Delta_T_tt)), eps=tol)

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
        discount_x, _ = tt_is_psd(new_X_tt, crit=crit)
        if discount_x:
            break
        else:
            new_X_tt[0][:, :, :, r:] *= discount
            x_step_size *= discount
    if active_ineq and discount_x:
        for iter in range(iters):
            discount_x, _ = tt_is_geq(lin_op_tt_ineq, new_X_tt, vec_bias_tt_ineq, crit=crit)
            if discount_x:
                break
            else:
                new_X_tt[0][:, :, :, r:] *= discount
                x_step_size *= discount

    r = Z_tt[0].shape[-1]
    new_Z_tt = tt_add(Z_tt, Delta_Z_tt)
    for iter in range(iters):
        discount_z, _ = tt_is_psd(new_Z_tt, crit=crit)
        if discount_z:
            break
        else:
            new_Z_tt[0][:, :, :, r:] *= discount
            z_step_size *= discount
    if active_ineq and discount_z:
        r = T_tt[0].shape[-1]
        new_T_tt = tt_add(T_tt, tt_scale(z_step_size, Delta_T_tt))
        for iter in range(iters):
            discount_z, _ = tt_is_geq_(new_T_tt, crit=crit, degenerate=True)
            if discount_z:
                break
            else:
                new_T_tt[0][:, :, :, r:] *= discount
                z_step_size *= discount
    return discount_x*x_step_size, discount_z*z_step_size


def tt_ipm(
    lag_maps,
    obj_tt,
    lin_op_tt,
    bias_tt,
    lin_op_tt_ineq=None,
    bias_tt_ineq=None,
    max_iter=100,
    feasibility_tol=1e-5,
    centrality_tol=1e-3,
    verbose=False,
    eps=1e-10
):
    dim = len(obj_tt)
    feasibility_tol = feasibility_tol / np.sqrt(dim)
    centrality_tol = centrality_tol / np.sqrt(dim)
    op_tol = 0.5*min(feasibility_tol, centrality_tol)
    lag_maps = {key: tt_rank_reduce(value, eps=op_tol) for key, value in lag_maps.items()}
    active_ineq = lin_op_tt_ineq is not None or bias_tt_ineq is not None
    num_blocks = 4 if active_ineq else 3
    local_solver = lambda prev_sol, lhs, rhs, local_auxs: ipm_solve_local_system(prev_sol, lhs, rhs, local_auxs, eps=eps, num_blocks=num_blocks)
    obj_tt = tt_rank_reduce(tt_vec(obj_tt), eps=op_tol)
    bias_tt = tt_rank_reduce(tt_vec(bias_tt), eps=op_tol)
    lhs_skeleton = {}
    lin_op_tt_adj = tt_transpose(lin_op_tt)
    lhs_skeleton[(0, 1)] = tt_rank_reduce(tt_scale(-1, lin_op_tt_adj), eps=op_tol)
    lhs_skeleton[(1, 0)] = tt_rank_reduce(tt_scale(-1, lin_op_tt), eps=op_tol)
    lin_op_tt_ineq_adj = None
    if active_ineq:
        lin_op_tt_ineq_adj = tt_scale(-1, tt_transpose(lin_op_tt_ineq))
        lhs_skeleton[(0, 2)] = tt_rank_reduce(tt_scale(-1, lin_op_tt_ineq_adj), eps=op_tol)
        lhs_skeleton[(0, 3)] = tt_identity(2*dim)
        bias_tt_ineq = tt_rank_reduce(tt_vec(bias_tt_ineq), eps=op_tol)
    else:
        lhs_skeleton[(0, 2)] = tt_identity(2 * dim)
    X_tt = tt_identity(dim)
    vec_Y_tt = [np.zeros((1, 2, 1)) for _ in range(2*dim)]
    T_tt = tt_one_matrix(dim)
    if active_ineq:
        T_tt = tt_mat(tt_fast_matrix_vec_mul(lin_op_tt_ineq_adj, tt_vec(T_tt), eps))
    Z_tt = tt_identity(dim)
    iter = 0
    for iter in range(1, max_iter):
        X_tt, vec_Y_tt, T_tt, Z_tt, pd_error, mu = _tt_ipm_newton_step(
            lag_maps,
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
            local_solver,
            verbose,
            eps
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
