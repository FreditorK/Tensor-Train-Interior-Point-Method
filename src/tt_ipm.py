import sys
import os

import scipy.linalg

sys.path.append(os.getcwd() + '/../')

from src.tt_ops import *
from src.tt_amen import tt_block_mals, _block_local_product
from src.tt_ineq_check import tt_pd_optimal_step_size

def forward_backward_sub(L, b):
    y = scip.linalg.solve_triangular(L, b, lower=True, check_finite=False)
    x = scip.linalg.solve_triangular(L.T, y, lower=False, check_finite=False, overwrite_b=True)
    return x


def approximate(M, k):
    # Step 1: Compute eigenvalues and eigenvectors
    eigvals, eigvecs = scip.sparse.linalg.eigsh(M, k, which="LM")  # For symmetric M
    idx = np.argsort(np.abs(eigvals))[::-1]  # Sort by absolute magnitude
    eigvals = eigvals[idx]
    eigvecs = eigvecs[:, idx]

    #lambda_k = eigvals[:k]  # Top-k eigenvalues
    V_k = eigvecs[:, :k]  # Top-k eigenvectors
    t = (np.trace(M) - np.sum(eigvals[:k]))/M.shape[0]
    M_approx = t*np.eye(M.shape[1]) + V_k @ (np.diag(eigvals[:k]) - t*np.eye(k)) @ V_k.T

    return M_approx

def _block_diag_local_product(XAX_k, block_A_k, XAX_kp1, x_core, blocks):
    result = np.zeros_like(x_core)
    for (i, j) in blocks:
        result[:, i % x_core.shape[1]] += einsum('lsr,smnS,LSR,rnR->lmL', XAX_k[(i, j)], block_A_k[(i, j)], XAX_kp1[(i, j)], x_core[:, j % x_core.shape[1]],  optimize="greedy")
    return result

def _ipm_m_free_approx_inv(XAX_k, block_A_k, XAX_k1, x_shape, blocks, k):
    block_size = x_shape[0]
    m = np.prod(x_shape[1:])

    def mat_vec(x_vec):
        return np.transpose(_block_diag_local_product(
            XAX_k, block_A_k, XAX_k1,
            np.transpose(x_vec.reshape(*x_shape), (1, 0, 2, 3)),
            blocks
        ), (1, 0, 2, 3)).reshape(-1, 1)

    linear_op = scip.sparse.linalg.LinearOperator((block_size * m, block_size * m), matvec=mat_vec)

    eigvals, eigvecs = scip.sparse.linalg.eigsh(linear_op, k, which="LM")  # For symmetric M
    min_eigval, _ = scip.sparse.linalg.eigsh(linear_op, 1, which="SA")
    idx = np.argsort(np.abs(eigvals))[::-1]  # Sort by absolute magnitude
    eigvals = eigvals[idx]
    eigvecs = eigvecs[:, idx]

    lambda_k = eigvals[:k]  # Top-k eigenvalues
    V_k = eigvecs[:, :k]  # Top-k eigenvectors
    t = (min_eigval + eigvals[k-1])/2
    V_k = V_k @ np.diag(np.sqrt(lambda_k))

    S_inv = np.linalg.inv(t*np.eye(k) + V_k.T @ V_k)

    def block_inv_approx(x):
        return (1/t)*(x - V_k @ (S_inv @ (V_k.T @ x)))

    return block_inv_approx


def _ipm_block_local_product(XAX_k, block_A_k, XAX_kp1, x_core, inv_I):
    result = np.zeros_like(x_core)
    result[0] = (
            einsum('lsr,smnS,LSR,rnR->lmL', XAX_k[(0, 0)], block_A_k[(0, 0)], XAX_kp1[(0, 0)], x_core[0], optimize="greedy") # K y
            + einsum('lsr,smnS,LSR,rnR->lmL', XAX_k[(0, 1)], block_A_k[(0, 1)], XAX_kp1[(0, 1)], x_core[1], optimize="greedy") # -L x
    )
    result[1] = -inv_I*einsum('lsr,smnS,LSR,rnR->lmL', XAX_k[(1, 0)], block_A_k[(1, 0)], XAX_kp1[(1, 0)], x_core[0], optimize="greedy") # invI*L^* y
    result[1] = einsum('lsr,smnS,LSR,rnR->lmL', XAX_k[(2, 2)], block_A_k[(2, 2)], XAX_kp1[(2, 2)], result[1], optimize="greedy") # L_X
    result[1] += einsum('lsr,smnS,LSR,rnR->lmL', XAX_k[(2, 1)], block_A_k[(2, 1)], XAX_kp1[(2, 1)], x_core[1], optimize="greedy") # L_Z x
    return result

def _ipm_local_solver(XAX_k, block_A_k, XAX_k1, Xb_k, block_b_k, Xb_k1, previous_solution, nrmsc, size_limit, rtol):
    x_shape = previous_solution.shape
    m = x_shape[0] * x_shape[2] * x_shape[3]
    rhs = np.zeros_like(previous_solution)
    rhs[:, 0] = einsum('br,bmB,BR->rmR', Xb_k[0], nrmsc * block_b_k[0], Xb_k1[0], optimize="greedy") if 0 in block_b_k else 0
    rhs[:, 1] = einsum('br,bmB,BR->rmR', Xb_k[1], nrmsc * block_b_k[1], Xb_k1[1], optimize="greedy") if 1 in block_b_k else 0
    rhs[:, 2] = einsum('br,bmB,BR->rmR', Xb_k[2], nrmsc * block_b_k[2], Xb_k1[2], optimize="greedy") if 2 in block_b_k else 0
    norm_rhs = np.linalg.norm(rhs)
    block_res_old = np.linalg.norm(_block_local_product(XAX_k, block_A_k, XAX_k1, previous_solution) - rhs) / norm_rhs
    if block_res_old < rtol:
        return previous_solution, block_res_old, block_res_old, rhs, norm_rhs
    if m <= size_limit:
        mR_p = rhs[:, 0].reshape(m, 1)
        mR_d = rhs[:, 1].reshape(m, 1)
        mR_c = rhs[:, 2].reshape(m, 1)
        L_L_Z = scip.linalg.cholesky(
            einsum('lsr,smnS,LSR->lmLrnR', XAX_k[(2, 1)], block_A_k[(2, 1)], XAX_k1[(2, 1)], optimize="greedy").reshape(
                m, m),
            check_finite=False, lower=True, overwrite_a=True
        )
        L_X = einsum('lsr,smnS,LSR->lmLrnR', XAX_k[(2, 2)], block_A_k[(2, 2)], XAX_k1[(2, 2)], optimize="greedy").reshape(m, m)
        mL_eq = einsum('lsr,smnS,LSR->lmLrnR', XAX_k[(0, 1)], block_A_k[(0, 1)], XAX_k1[(0, 1)], optimize="greedy").reshape(m, m)
        mL_eq_adj = einsum('lsr,smnS,LSR->lmLrnR', XAX_k[(1, 0)], block_A_k[(1, 0)], XAX_k1[(1, 0)], optimize="greedy").reshape(m, m)
        inv_I = np.divide(1, einsum('lsr,smnS,LSR->lmL', XAX_k[(1, 2)], block_A_k[(1, 2)], XAX_k1[(1, 2)], optimize="greedy").reshape(1, -1))
        A = mL_eq @ forward_backward_sub(L_L_Z, L_X * inv_I) @ mL_eq_adj + einsum('lsr,smnS,LSR->lmLrnR', XAX_k[(0, 0)], block_A_k[(0, 0)], XAX_k1[(0, 0)], optimize="greedy").reshape(m, m)
        b = mR_p - mL_eq @ forward_backward_sub(L_L_Z, mR_c - (L_X * inv_I.reshape(1, -1)) @ mR_d) - A @ np.transpose(previous_solution, (1, 0, 2, 3))[1].reshape(-1, 1)
        y = scip.linalg.solve(A, b, check_finite=False, overwrite_a=True, overwrite_b=True) + np.transpose(previous_solution, (1, 0, 2, 3))[1].reshape(-1, 1)
        z = inv_I.reshape(-1, 1) * (mR_d - mL_eq_adj @ y)
        x = forward_backward_sub(L_L_Z, mR_c - L_X @ z)
        solution_now = np.transpose(np.vstack((y, x, z)).reshape(x_shape[1], x_shape[0], x_shape[2], x_shape[3]), (1, 0, 2, 3))
    else:
        inv_I = np.divide(1, einsum('lsr,smnS,LSR->lmL', XAX_k[(1, 2)], block_A_k[(1, 2)], XAX_k1[(1, 2)], optimize="greedy"))
        def mat_vec(x_vec):
            x_vec = _ipm_block_local_product(
                XAX_k, block_A_k, XAX_k1,
                x_vec.reshape(2, x_shape[0], x_shape[2], x_shape[3]), inv_I
            ).reshape(-1, 1)
            return x_vec

        linear_op = scip.sparse.linalg.LinearOperator((2 * m, 2 * m), matvec=mat_vec)
        local_rhs = -_ipm_block_local_product(XAX_k, block_A_k, XAX_k1, np.transpose(previous_solution[:, :2], (1, 0, 2, 3)), inv_I)
        local_rhs[0] += rhs[:, 0]
        local_rhs[1] += rhs[:, 2] - einsum('lsr,smnS,LSR,rnR->lmL', XAX_k[(2, 2)], block_A_k[(2, 2)], XAX_k1[(2, 2)], inv_I*rhs[:, 1], optimize="greedy")
        solution_now, _ = scip.sparse.linalg.bicgstab(
            linear_op,
            local_rhs.reshape(-1, 1),
            rtol=1e-3*block_res_old,
            maxiter=100
        )
        solution_now = np.transpose(solution_now.reshape(2, x_shape[0], x_shape[2], x_shape[3]), (1, 0, 2, 3)) + previous_solution[:, :2]
        z = inv_I * (rhs[:, 1] - einsum('lsr,smnS,LSR,rnR->lmL', XAX_k[(1, 0)], block_A_k[(1, 0)], XAX_k1[(1, 0)], solution_now[:, 0], optimize="greedy"))
        solution_now = np.concatenate((solution_now, z.reshape(x_shape[0], 1, x_shape[2], x_shape[3])), axis=1)

    block_res_new = np.linalg.norm(_block_local_product(XAX_k, block_A_k, XAX_k1, solution_now) - rhs) / norm_rhs

    if block_res_old < block_res_new:
        solution_now = previous_solution

    return solution_now, block_res_old, min(block_res_new, block_res_old), rhs, norm_rhs

def tt_infeasible_newton_system(
        lhs_skeleton,
        obj_tt,
        X_tt,
        Y_tt,
        Z_tt,
        T_tt,
        lin_op_tt,
        lin_op_tt_adj,
        bias_tt,
        lin_op_tt_ineq,
        lin_op_tt_ineq_adj,
        bias_tt_ineq,
        op_tol,
        feasibility_tol,
        centrality_done,
        active_ineq,
        direction,
        eps
):
    idx_add = int(active_ineq)
    P = tt_identity(len(Z_tt))
    if direction == "XZ":
        L_Z = tt_rank_reduce(tt_kron(Z_tt, P), eps=op_tol, rank_weighted_error=True)
        L_X = tt_rank_reduce(tt_kron(P, X_tt), eps=op_tol, rank_weighted_error=True)
    else:
        L_Z = tt_rank_reduce(tt_scale(0.5, tt_add(tt_kron(P, Z_tt), tt_kron(Z_tt, P))), eps=op_tol,
                             rank_weighted_error=True)
        L_X = tt_rank_reduce(tt_scale(0.5, tt_add(tt_kron(X_tt, P), tt_kron(P, X_tt))), eps=op_tol,
                             rank_weighted_error=True)

    X_tt = tt_reshape(X_tt, (4, ))
    Y_tt = tt_reshape(Y_tt, (4, ))
    Z_tt = tt_reshape(Z_tt, (4, ))

    if active_ineq:
        ineq_res_tt = tt_sub(bias_tt_ineq, tt_fast_matrix_vec_mul(lin_op_tt_ineq, X_tt, eps))
        mat_ineq_res_op_tt = tt_diag(ineq_res_tt)
        lhs_skeleton[(2, 2)] = tt_rank_reduce(mat_ineq_res_op_tt, op_tol, rank_weighted_error=True)
        Tmat_lin_op_tt_ineq =  tt_fast_mat_mat_mul(tt_reshape(tt_diag(tt_vec(T_tt)), (4, 4)), lin_op_tt_ineq, eps)
        lhs_skeleton[(2, 0)] = tt_rank_reduce(tt_scale(-1, Tmat_lin_op_tt_ineq), eps=op_tol, rank_weighted_error=True)
    lhs_skeleton[(2 + idx_add, 1)] = L_Z
    lhs_skeleton[(2 + idx_add, 2 + idx_add)] = L_X

    rhs = {}
    dual_feas = tt_sub(tt_fast_matrix_vec_mul(lin_op_tt_adj, Y_tt, eps), tt_rank_reduce(tt_add(Z_tt, obj_tt), eps, rank_weighted_error=True))
    primal_feas = tt_rank_reduce(tt_sub(tt_fast_matrix_vec_mul(lin_op_tt, X_tt, eps), bias_tt), op_tol, rank_weighted_error=True)  # primal feasibility
    primal_error = tt_inner_prod(primal_feas, primal_feas)
    if primal_error > 0.5*feasibility_tol:
        rhs[0] = primal_feas

    if active_ineq:
        reshaped_T = tt_reshape(T_tt, (4, ))
        dual_feas = tt_add(dual_feas, tt_fast_matrix_vec_mul(lin_op_tt_ineq_adj, reshaped_T, eps))
        # TODO: Does mu 1 not also be under mat_lin_op_tt_ineq, need to adjust mu 1 to have zeros where L(X) has zeros
        one = tt_fast_matrix_vec_mul(lin_op_tt_ineq_adj, [np.ones((1, 4, 1)) for _ in X_tt])
        Tineq_res_tt = tt_fast_hadammard(reshaped_T, ineq_res_tt, eps)
        nu = max(2*sigma*tt_inner_prod([0.5*c for c in reshaped_T], ineq_res_tt), 0.5*op_tol)
        primal_feas_ineq = tt_rank_reduce(tt_sub(tt_scale(nu, one), Tineq_res_tt), 0.5 * op_tol, rank_weighted_error=True)
        #primal_ineq_error = tt_inner_prod(primal_feas_ineq, primal_feas_ineq)
        rhs[2] = primal_feas_ineq

    dual_feas = tt_rank_reduce(dual_feas, op_tol, rank_weighted_error=True)
    dual_error = tt_inner_prod(dual_feas, dual_feas)
    if dual_error > 0.5*feasibility_tol:
        rhs[1] = dual_feas

    if not centrality_done:
        XZ_term = tt_fast_matrix_vec_mul(L_X, Z_tt, eps)
        rhs[2 + idx_add] = tt_rank_reduce(tt_scale(-1, XZ_term), op_tol, rank_weighted_error=True)

    return lhs_skeleton, rhs, (primal_error, dual_error)

def _tt_symmetrise(matrix_tt, err_bound):
    return tt_rank_reduce(tt_scale(0.5, tt_add(matrix_tt, tt_transpose(matrix_tt))), eps=err_bound, rank_weighted_error=True)


def _tt_get_block(i, block_matrix_tt):
    if len(block_matrix_tt[0].shape) < len(block_matrix_tt[-1].shape):
        return block_matrix_tt[:-1] + [block_matrix_tt[-1][:, i]]
    return [block_matrix_tt[0][:, i]] + block_matrix_tt[1:]

def _tt_ipm_newton_step(
            vec_obj_tt,
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
            op_tol,
            feasibility_tol,
            centrality_tol,
            eps,
            active_ineq,
            solver,
            direction,
            verbose
):
    dim = len(Z_tt)
    ZX = tt_inner_prod(Z_tt, X_tt)
    mu = np.divide(ZX, 2**dim)
    centrality_done = np.less(mu, centrality_tol)
    lhs_matrix_tt, rhs_vec_tt, primal_dual_error = tt_infeasible_newton_system(
        lhs_skeleton,
        vec_obj_tt,
        X_tt,
        Y_tt,
        Z_tt,
        T_tt,
        lin_op_tt,
        lin_op_tt_adj,
        bias_tt,
        lin_op_tt_ineq,
        lin_op_tt_ineq_adj,
        bias_tt_ineq,
        op_tol,
        feasibility_tol,
        centrality_done,
        active_ineq,
        direction,
        eps
    )
    if centrality_done and np.less(primal_dual_error[0], feasibility_tol) and np.less(primal_dual_error[1], feasibility_tol):
        return 0, 0, None, None, None, primal_dual_error, mu, 1
    # Predictor
    if verbose:
        print("--- Predictor  step ---")
    idx_add = int(active_ineq)
    Delta_tt, res = solver(lhs_matrix_tt, rhs_vec_tt, None, 8)
    Delta_X_tt = tt_rank_reduce(tt_reshape(_tt_get_block(1, Delta_tt), (2, 2)), eps=op_tol, rank_weighted_error=True)
    Delta_Z_tt = tt_rank_reduce(tt_reshape(_tt_get_block(2 + idx_add, Delta_tt), (2, 2)), eps=op_tol, rank_weighted_error=True)
    Delta_X_tt = _tt_symmetrise(Delta_X_tt, op_tol)
    Delta_Z_tt = _tt_symmetrise(Delta_Z_tt, op_tol)

    x_step_size, z_step_size, _ = _tt_line_search(
        X_tt, Z_tt,
        Delta_X_tt, Delta_Z_tt,
        op_tol, eps
    )

    if (x_step_size < 1 or z_step_size < 1) and not centrality_done:
        # Corrector
        if verbose:
            print("\n--- Centering-Corrector  step ---")
        P = tt_identity(dim)
        if direction == "XZ":
            L_Z = tt_scale(-1, tt_rank_reduce(tt_kron(Delta_Z_tt, P), eps=op_tol, rank_weighted_error=True))
            # No rank increase for operators
        else:
            L_Z = tt_scale(-0.5, tt_rank_reduce(tt_add(tt_kron(P, Delta_Z_tt), tt_kron(Delta_Z_tt, P)), eps=op_tol,
                                                rank_weighted_error=True))
        sigma = ((ZX + x_step_size * z_step_size * tt_inner_prod(Delta_X_tt, Delta_Z_tt) + z_step_size * tt_inner_prod(
            X_tt, Delta_Z_tt) + x_step_size * tt_inner_prod(Delta_X_tt, Z_tt)) / ZX) ** 2
        rhs_vec_tt[2 + idx_add] = tt_rank_reduce(
            tt_add(
                tt_scale(sigma*mu, tt_reshape(tt_identity(len(X_tt)), (4, ))),
                tt_add(
                rhs_vec_tt[2 + idx_add],
                tt_fast_matrix_vec_mul(L_Z, tt_reshape(Delta_X_tt, (4, )), eps)
                )
            ),
            op_tol,
            rank_weighted_error=False
        )

        Delta_tt, res = solver(lhs_matrix_tt, rhs_vec_tt, Delta_tt, 6)
        Delta_X_tt = tt_rank_reduce(tt_reshape(_tt_get_block(1, Delta_tt), (2, 2)), eps=op_tol, rank_weighted_error=True)
        Delta_Z_tt = tt_rank_reduce(tt_reshape(_tt_get_block(2 + idx_add, Delta_tt), (2, 2)), eps=op_tol, rank_weighted_error=True)
        Delta_X_tt = _tt_symmetrise(Delta_X_tt, op_tol)
        Delta_Z_tt = _tt_symmetrise(Delta_Z_tt, op_tol)
        x_step_size, z_step_size, _ = _tt_line_search(
            X_tt, Z_tt,
            Delta_X_tt, Delta_Z_tt,
            op_tol, eps
        )
    else:
        sigma = None
    Delta_Y_tt = tt_rank_reduce(tt_reshape(_tt_get_block(0, Delta_tt), (2, 2)), eps=op_tol, rank_weighted_error=True)

    return x_step_size, z_step_size, Delta_X_tt, Delta_Y_tt, Delta_Z_tt, primal_dual_error, mu, sigma


def _tt_line_search(
        X_tt,
        Z_tt,
        Delta_X_tt,
        Delta_Z_tt,
        op_tol,
        eps
):
    x_step_size, permitted_err_x = tt_pd_optimal_step_size(X_tt, Delta_X_tt, op_tol, eps=eps)
    z_step_size, permitted_err_z = tt_pd_optimal_step_size(Z_tt, Delta_Z_tt, op_tol, eps=eps)
    if x_step_size == 1 and z_step_size == 1:
        return 1, 1, min(permitted_err_x, permitted_err_z)
    tau_x = 0.9 + 0.09*x_step_size
    tau_z = 0.9 + 0.09*z_step_size
    return tau_x*x_step_size, tau_z*z_step_size, min(permitted_err_x, permitted_err_z)


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
    op_tol=1e-5,
    verbose=False,
    direction = "XZ",
    eps=1e-12
):
    active_ineq = lin_op_tt_ineq is not None or bias_tt_ineq is not None
    dim = len(obj_tt)
    # Reshaping
    lag_maps = {key: tt_rank_reduce(tt_reshape(value, (4, 4)), eps=op_tol) for key, value in lag_maps.items()}
    obj_tt = tt_rank_reduce(tt_reshape(obj_tt, (4, )), eps=op_tol)
    lin_op_tt = tt_rank_reduce(tt_reshape(lin_op_tt, (4, 4)), eps=op_tol)
    bias_tt = tt_rank_reduce(tt_reshape(bias_tt, (4, )), eps=op_tol)
    # -------------
    # Normalisation
    scaling_factor = np.sqrt(dim)
    obj_tt = tt_normalise(obj_tt, radius=0.5*scaling_factor) # TODO: normalize by the trace of sol_X because Z approx= C in trace and magnitude, this gives better conditioning
    lag_maps = {key: tt_scale(scaling_factor, value) for key, value in lag_maps.items()}
    lin_op_tt = tt_scale(scaling_factor, lin_op_tt)
    bias_tt = tt_scale(scaling_factor, bias_tt)

    solver = lambda lhs, rhs, x0, nwsp: tt_block_mals(
        lhs,
        rhs,
        x0=x0,
        local_solver=_ipm_local_solver,
        tol=0.5*op_tol,
        nswp=nwsp,
        verbose=verbose,
        rank_weighted_error=False
    )
    lhs_skeleton = {}
    lin_op_tt_adj = tt_transpose(lin_op_tt)
    lhs_skeleton[(1, 0)] = tt_scale(-1, lin_op_tt_adj)
    lhs_skeleton[(0, 1)] = tt_scale(-1, lin_op_tt)
    lhs_skeleton[(0, 0)] = lag_maps["y"]
    lin_op_tt_ineq_adj = None
    if active_ineq:
        lin_op_tt_ineq_adj = tt_scale(-1, tt_transpose(lin_op_tt_ineq))
        lhs_skeleton[(0, 2)] = tt_rank_reduce(tt_reshape(tt_scale(-1, lin_op_tt_ineq_adj), (4, 4)), eps=op_tol)
        lhs_skeleton[(0, 3)] = tt_reshape(tt_identity(2 * dim), (4, 4))
        bias_tt_ineq = tt_rank_reduce(tt_reshape(bias_tt_ineq, (4, )), eps=op_tol)
    else:
        lhs_skeleton[(1, 2)] = tt_reshape(tt_identity(2 * dim), (4, 4))
    x_initial_step = tt_norm(bias_tt) / tt_norm(tt_diagonal(lin_op_tt))
    X_tt = tt_scale(x_initial_step, tt_identity(dim))
    Y_tt = tt_zero_matrix(2*dim)
    T_tt = None
    if active_ineq:
        T_tt = tt_reshape(tt_fast_matrix_vec_mul(lin_op_tt_ineq_adj, tt_reshape(tt_one_matrix(dim), (4, )), eps), (2, 2))
        T_tt = tt_normalise(T_tt)
    Z_tt = tt_identity(dim)
    Delta_Z_tt = tt_scale(-1, tt_reshape(obj_tt, (2, 2)))
    z_inital_step, _ = tt_pd_optimal_step_size(Z_tt, Delta_Z_tt, op_tol, eps=eps)
    Z_tt = tt_rank_reduce(tt_add(Z_tt, tt_scale(0.9*z_inital_step, Delta_Z_tt)), eps=op_tol, rank_weighted_error=True)
    iteration = 0
    for iteration in range(1, max_iter):
        x_step_size, z_step_size, Delta_X_tt, Delta_Y_tt, Delta_Z_tt, pd_error, mu, sigma = _tt_ipm_newton_step(
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
            op_tol,
            feasibility_tol,
            centrality_tol,
            eps,
            active_ineq,
            solver,
            direction,
            verbose
        )
        progress_percent = np.abs(mu)
        if x_step_size == 0 and z_step_size == 0:
            break
        X_tt = tt_rank_reduce(tt_add(X_tt, tt_scale(x_step_size, Delta_X_tt)), eps=op_tol, rank_weighted_error=True)
        Y_tt = tt_add(Y_tt, tt_scale(z_step_size, Delta_Y_tt))
        Y_tt = tt_rank_reduce(tt_sub(Y_tt, tt_reshape(tt_fast_matrix_vec_mul(lag_maps["y"], tt_reshape(Y_tt, shape=(4, )), eps), (2, 2))), eps=op_tol, rank_weighted_error=True)
        Z_tt = tt_rank_reduce(tt_add(Z_tt, tt_scale(z_step_size, Delta_Z_tt)), eps=op_tol, rank_weighted_error=True)
        if verbose:
            print(f"---Step {iteration}---")
            print(f"Step sizes: {x_step_size}, {z_step_size}")
            print(f"Duality Gap: {100*progress_percent:.4f}")
            print(f"Primal-Dual error: {pd_error}")
            print(f"Sigma: {sigma}")
            print(
                f"Ranks X_tt: {tt_ranks(X_tt)}, Z_tt: {tt_ranks(Z_tt)}, \n"
                f"      Y_tt: {tt_ranks(Y_tt)}, T_tt: {tt_ranks(T_tt) if active_ineq else None} \n"
            )
    print(f"---Terminated---")
    print(f"Converged in {iteration} iterations.")
    print(f"Duality Gap: {100 * progress_percent:.4f}")
    print(f"Primal-Dual error: {pd_error}")
    print(
        f"Ranks X_tt: {tt_ranks(X_tt)}, Z_tt: {tt_ranks(Z_tt)}, \n"
        f"      Y_tt: {tt_ranks(Y_tt)}, T_tt: {tt_ranks(T_tt) if active_ineq else None} \n"
    )
    return X_tt, Y_tt, T_tt, Z_tt, {"num_iters": iteration}