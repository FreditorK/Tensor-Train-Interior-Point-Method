import sys
import os

sys.path.append(os.getcwd() + '/../')

from src.tt_ops import *
from src.tt_amen import tt_block_mals, _block_local_product, cached_einsum
from src.tt_ineq_check import tt_pd_optimal_step_size

def forward_backward_sub(L, b):
    y = scip.linalg.solve_triangular(L, b, lower=True, check_finite=False)
    x = scip.linalg.solve_triangular(L.T, y, lower=False, check_finite=False, overwrite_b=True)
    return x

def _ipm_block_local_product(XAX_k, block_A_k, XAX_kp1, x_core, inv_I):
    result = np.zeros_like(x_core)
    result[0] = (
            cached_einsum('lsr,smnS,LSR,rnR->lmL', XAX_k[(0, 0)], block_A_k[(0, 0)], XAX_kp1[(0, 0)], x_core[0]) # K y
            + cached_einsum('lsr,smnS,LSR,rnR->lmL', XAX_k[(0, 1)], block_A_k[(0, 1)], XAX_kp1[(0, 1)], x_core[1]) # -L x
    )
    result[1] = -inv_I*cached_einsum('lsr,smnS,LSR,rnR->lmL', XAX_k[(1, 0)], block_A_k[(1, 0)], XAX_kp1[(1, 0)], x_core[0]) # invI*L^* y
    result[1] = cached_einsum('lsr,smnS,LSR,rnR->lmL', XAX_k[(2, 2)], block_A_k[(2, 2)], XAX_kp1[(2, 2)], result[1]) # L_X
    result[1] += cached_einsum('lsr,smnS,LSR,rnR->lmL', XAX_k[(2, 1)], block_A_k[(2, 1)], XAX_kp1[(2, 1)], x_core[1]) # L_Z x
    return result

def _ipm_local_solver(XAX_k, block_A_k, XAX_k1, Xb_k, block_b_k, Xb_k1, previous_solution, nrmsc, size_limit, rtol):
    x_shape = previous_solution.shape
    m = x_shape[0] * x_shape[2] * x_shape[3]
    rhs = np.zeros_like(previous_solution)
    rhs[:, 0] = cached_einsum('br,bmB,BR->rmR', Xb_k[0], nrmsc * block_b_k[0], Xb_k1[0]) if 0 in block_b_k else 0
    rhs[:, 1] = cached_einsum('br,bmB,BR->rmR', Xb_k[1], nrmsc * block_b_k[1], Xb_k1[1]) if 1 in block_b_k else 0
    rhs[:, 2] = cached_einsum('br,bmB,BR->rmR', Xb_k[2], nrmsc * block_b_k[2], Xb_k1[2]) if 2 in block_b_k else 0
    inv_I = np.divide(1, cached_einsum('lsr,smnS,LSR->lmL', XAX_k[(1, 2)], block_A_k[(1, 2)], XAX_k1[(1, 2)]))
    norm_rhs = np.linalg.norm(rhs)
    block_res_old = np.linalg.norm(_block_local_product(XAX_k, block_A_k, XAX_k1, previous_solution) - rhs) / norm_rhs
    if block_res_old < rtol:
        return previous_solution, block_res_old, block_res_old, rhs, norm_rhs
    if m <= size_limit:
        mR_p = rhs[:, 0].reshape(m, 1)
        mR_d = rhs[:, 1].reshape(m, 1)
        mR_c = rhs[:, 2].reshape(m, 1)
        try:
            L_L_Z = scip.linalg.cholesky(
                cached_einsum('lsr,smnS,LSR->lmLrnR', XAX_k[(2, 1)], block_A_k[(2, 1)], XAX_k1[(2, 1)]).reshape(m, m),
                check_finite=False, lower=True, overwrite_a=True
            )
            L_X = cached_einsum('lsr,smnS,LSR->lmLrnR', XAX_k[(2, 2)], block_A_k[(2, 2)], XAX_k1[(2, 2)]).reshape(m, m)
            mL_eq = cached_einsum('lsr,smnS,LSR->lmLrnR', XAX_k[(0, 1)], block_A_k[(0, 1)], XAX_k1[(0, 1)]).reshape(m, m)
            mL_eq_adj = cached_einsum('lsr,smnS,LSR->lmLrnR', XAX_k[(1, 0)], block_A_k[(1, 0)], XAX_k1[(1, 0)]).reshape(m, m)
            inv_I = inv_I.reshape(1, -1)
            A = mL_eq @ forward_backward_sub(L_L_Z, L_X * inv_I) @ mL_eq_adj + cached_einsum('lsr,smnS,LSR->lmLrnR', XAX_k[(0, 0)], block_A_k[(0, 0)], XAX_k1[(0, 0)]).reshape(m, m)
            b = mR_p - mL_eq @ forward_backward_sub(L_L_Z, mR_c - (L_X * inv_I.reshape(1, -1)) @ mR_d) - A @ np.transpose(previous_solution, (1, 0, 2, 3))[1].reshape(-1, 1)
            y = scip.linalg.solve(A, b, check_finite=False, overwrite_a=True, overwrite_b=True) + np.transpose(previous_solution, (1, 0, 2, 3))[1].reshape(-1, 1)
            z = inv_I.reshape(-1, 1) * (mR_d - mL_eq_adj @ y)
            x = forward_backward_sub(L_L_Z, mR_c - L_X @ z)
            solution_now = np.transpose(np.vstack((y, x, z)).reshape(x_shape[1], x_shape[0], x_shape[2], x_shape[3]), (1, 0, 2, 3))
        except:
            size_limit = 0

    if m > size_limit:
        def mat_vec(x_vec):
            x_vec = _ipm_block_local_product(
                XAX_k, block_A_k, XAX_k1,
                x_vec.reshape(2, x_shape[0], x_shape[2], x_shape[3]), inv_I
            ).reshape(-1, 1)
            return x_vec

        linear_op = scip.sparse.linalg.LinearOperator((2 * m, 2 * m), matvec=mat_vec)
        local_rhs = -_ipm_block_local_product(XAX_k, block_A_k, XAX_k1, np.transpose(previous_solution[:, :2], (1, 0, 2, 3)), inv_I)
        local_rhs[0] += rhs[:, 0]
        local_rhs[1] += rhs[:, 2] - cached_einsum('lsr,smnS,LSR,rnR->lmL', XAX_k[(2, 2)], block_A_k[(2, 2)], XAX_k1[(2, 2)], inv_I*rhs[:, 1])
        solution_now, _ = scip.sparse.linalg.bicgstab(
            linear_op,
            local_rhs.reshape(-1, 1),
            rtol=1e-3*block_res_old,
            maxiter=50
        )
        solution_now = np.transpose(solution_now.reshape(2, x_shape[0], x_shape[2], x_shape[3]), (1, 0, 2, 3)) + previous_solution[:, :2]
        z = inv_I * (rhs[:, 1] - cached_einsum('lsr,smnS,LSR,rnR->lmL', XAX_k[(1, 0)], block_A_k[(1, 0)], XAX_k1[(1, 0)], solution_now[:, 0]))
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
        ineq_mask,
        op_tol,
        feasibility_tol,
        centrality_done,
        direction,
        eps
):
    P = tt_identity(len(Z_tt))
    rhs = {}

    if direction == "XZ":
        L_Z = tt_rank_reduce(tt_kron(Z_tt, P), eps=op_tol, rank_weighted_error=True)
        L_X = tt_rank_reduce(tt_kron(P, X_tt), eps=op_tol, rank_weighted_error=True)
        if not centrality_done:
            rhs[2] = tt_reshape(tt_scale(-1, tt_rank_reduce(tt_fast_mat_mat_mul(Z_tt, X_tt, eps), eps=op_tol, rank_weighted_error=True)), (4, ))
    else:
        L_Z = tt_rank_reduce(tt_scale(0.5, tt_add(tt_kron(P, Z_tt), tt_kron(Z_tt, P))), eps=op_tol,
                             rank_weighted_error=True)
        L_X = tt_rank_reduce(tt_scale(0.5, tt_add(tt_kron(X_tt, P), tt_kron(P, X_tt))), eps=op_tol,
                             rank_weighted_error=True)
        if not centrality_done:
            rhs[2] = tt_reshape(tt_scale(-1, _tt_symmetrise(tt_fast_mat_mat_mul(X_tt, Z_tt, 0.5*op_tol), 0.5*op_tol)), (4, ))

    if ineq_mask is not None:
        lhs_skeleton[(3, 1)] =  tt_rank_reduce(tt_diag(tt_vec(T_tt)), eps=op_tol, rank_weighted_error=True)
        lhs_skeleton[(3, 3)] = tt_rank_reduce(tt_add(lhs_skeleton[(3, 3)], tt_diag(tt_vec(X_tt))), eps=op_tol, rank_weighted_error=True)
    lhs_skeleton[(2, 1)] = L_Z
    lhs_skeleton[(2, 2)] = L_X

    X_tt = tt_reshape(X_tt, (4, ))
    Y_tt = tt_reshape(Y_tt, (4, ))
    Z_tt = tt_reshape(Z_tt, (4, ))

    dual_feas = tt_rank_reduce(tt_sub(tt_fast_matrix_vec_mul(lin_op_tt_adj, Y_tt, eps), tt_rank_reduce(tt_add(Z_tt, obj_tt), eps, rank_weighted_error=True)), op_tol, rank_weighted_error=True)
    primal_feas = tt_rank_reduce(tt_sub(tt_fast_matrix_vec_mul(lin_op_tt, X_tt, eps), bias_tt), op_tol, rank_weighted_error=True)  # primal feasibility
    primal_error = tt_inner_prod(primal_feas, primal_feas)
    if primal_error > 0.5*feasibility_tol:
        rhs[0] = primal_feas

    if ineq_mask is not None:
        dual_feas = tt_rank_reduce(tt_add(dual_feas, tt_reshape(T_tt, (4, ))), op_tol, rank_weighted_error=True)

    dual_error = tt_inner_prod(dual_feas, dual_feas)
    if dual_error > 0.5*feasibility_tol:
        rhs[1] = dual_feas

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
            ineq_mask,
            X_tt,
            Y_tt,
            Z_tt,
            T_tt,
            op_tol,
            feasibility_tol,
            centrality_tol,
            eps,
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
        ineq_mask,
        op_tol,
        feasibility_tol,
        centrality_done,
        direction,
        eps
    )
    if centrality_done and np.less(primal_dual_error[0], feasibility_tol) and np.less(primal_dual_error[1], feasibility_tol):
        if ineq_mask is not None:
            ineq_mask = None
        else:
            return 0, 0, None, None, None, None, primal_dual_error, mu, 1
    # Predictor
    if verbose:
        print("--- Predictor  step ---")
    Delta_tt, res = solver(lhs_matrix_tt, rhs_vec_tt, None, 8)
    Delta_X_tt = tt_rank_reduce(tt_reshape(_tt_get_block(1, Delta_tt), (2, 2)), eps=op_tol, rank_weighted_error=True)
    Delta_Z_tt = tt_rank_reduce(tt_reshape(_tt_get_block(2, Delta_tt), (2, 2)), eps=op_tol, rank_weighted_error=True)
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
        if direction == "XZ":
            Delta_XZ_term = tt_scale(-1, tt_fast_mat_mat_mul(Delta_Z_tt, Delta_X_tt, op_tol))
        else:
            Delta_XZ_term = tt_scale(-1, _tt_symmetrise(tt_fast_mat_mat_mul(Delta_X_tt, Delta_Z_tt, 0.5*op_tol), 0.5*op_tol))

        sigma = ((ZX + x_step_size * z_step_size * tt_inner_prod(Delta_X_tt, Delta_Z_tt) + z_step_size * tt_inner_prod(
            X_tt, Delta_Z_tt) + x_step_size * tt_inner_prod(Delta_X_tt, Z_tt)) / ZX) ** 2
        rhs_vec_tt[2] = tt_rank_reduce(
            tt_add(
                tt_scale(sigma*mu, tt_reshape(tt_identity(len(X_tt)), (4, ))),
                tt_add(
                rhs_vec_tt[2],
                tt_reshape(Delta_XZ_term, (4, ))
                )
            ),
            op_tol,
            rank_weighted_error=True
        )

        Delta_tt, res = solver(lhs_matrix_tt, rhs_vec_tt, Delta_tt, 6)
        Delta_X_tt = tt_rank_reduce(tt_reshape(_tt_get_block(1, Delta_tt), (2, 2)), eps=op_tol, rank_weighted_error=True)
        Delta_Z_tt = tt_rank_reduce(tt_reshape(_tt_get_block(2, Delta_tt), (2, 2)), eps=op_tol, rank_weighted_error=True)
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
    Delta_T_tt = None
    if ineq_mask is not None:
        Delta_T_tt = tt_rank_reduce(tt_reshape(_tt_get_block(3, Delta_tt), (2, 2)), eps=op_tol, rank_weighted_error=True)

    return x_step_size, z_step_size, Delta_X_tt, Delta_Y_tt, Delta_Z_tt, Delta_T_tt, primal_dual_error, mu, sigma


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
    ineq_mask=None,
    max_iter=100,
    feasibility_tol=1e-5,
    centrality_tol=1e-3,
    op_tol=1e-5,
    verbose=False,
    direction = "XZ",
    eps=1e-12
):
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
        tol=0.5*min(feasibility_tol, centrality_tol),
        nswp=nwsp,
        verbose=verbose,
        rank_weighted_error=False
    )
    lhs_skeleton = {}
    lin_op_tt_adj = tt_transpose(lin_op_tt)
    lhs_skeleton[(1, 0)] = tt_scale(-1, lin_op_tt_adj)
    lhs_skeleton[(0, 1)] = tt_scale(-1, lin_op_tt)
    lhs_skeleton[(0, 0)] = lag_maps["y"]
    lhs_skeleton[(1, 2)] = tt_reshape(tt_identity(2 * dim), (4, 4))
    if ineq_mask is not None:
        lhs_skeleton[(3, 3)] = lag_maps["t"]
        lhs_skeleton[(1, 3)] = tt_reshape(tt_identity(2 * dim), (4, 4))
    x_initial_step = tt_norm(bias_tt) / tt_norm(tt_diagonal(lin_op_tt))
    X_tt = tt_scale(x_initial_step, tt_identity(dim))
    Y_tt = tt_zero_matrix(2*dim)
    Z_tt = tt_identity(dim)
    Delta_Z_tt = tt_scale(-1, tt_reshape(obj_tt, (2, 2)))
    z_inital_step, _ = tt_pd_optimal_step_size(Z_tt, Delta_Z_tt, op_tol, eps=eps)
    Z_tt = tt_rank_reduce(tt_add(Z_tt, tt_scale(0.9*z_inital_step, Delta_Z_tt)), eps=op_tol, rank_weighted_error=True)
    T_tt = None
    if ineq_mask is not None:
        T_tt = tt_rank_reduce(ineq_mask, eps=op_tol, rank_weighted_error=True)
    iteration = 0
    for iteration in range(1, max_iter):
        x_step_size, z_step_size, Delta_X_tt, Delta_Y_tt, Delta_Z_tt, Delta_T_tt, pd_error, mu, sigma = _tt_ipm_newton_step(
            obj_tt,
            lhs_skeleton,
            lin_op_tt,
            lin_op_tt_adj,
            bias_tt,
            ineq_mask,
            X_tt,
            Y_tt,
            Z_tt,
            T_tt,
            op_tol,
            feasibility_tol,
            centrality_tol,
            eps,
            solver,
            direction,
            verbose
        )
        progress_percent = np.abs(mu)
        if x_step_size == 0 and z_step_size == 0:
            break
        X_tt = tt_rank_reduce(tt_add(X_tt, tt_scale(x_step_size, Delta_X_tt)), eps=op_tol, rank_weighted_error=True)
        Y_tt =  tt_rank_reduce(tt_add(Y_tt, tt_scale(z_step_size, Delta_Y_tt)), eps=0.5*op_tol, rank_weighted_error=True)
        Y_tt = tt_rank_reduce(tt_sub(Y_tt, tt_reshape(tt_fast_matrix_vec_mul(lag_maps["y"], tt_reshape(Y_tt, shape=(4, )), eps), (2, 2))), eps=0.5*op_tol, rank_weighted_error=True)
        Z_tt = tt_rank_reduce(tt_add(Z_tt, tt_scale(z_step_size, Delta_Z_tt)), eps=op_tol, rank_weighted_error=True)
        if verbose:
            print(f"---Step {iteration}---")
            print(f"Step sizes: {x_step_size:.4f}, {z_step_size:.4f}")
            print(f"Duality Gap: {100*progress_percent:.4f}")
            print(f"Primal-Dual error: {pd_error}")
            print(f"Sigma: {sigma}")
            print(
                f"Ranks X_tt: {tt_ranks(X_tt)}, Z_tt: {tt_ranks(Z_tt)}, \n"
                f"      Y_tt: {tt_ranks(Y_tt)}, T_tt: {tt_ranks(T_tt) if T_tt else None} \n"
            )
    print(f"---Terminated---")
    print(f"Converged in {iteration} iterations.")
    print(f"Duality Gap: {100 * progress_percent:.4f}")
    print(f"Primal-Dual error: {pd_error}")
    print(
        f"Ranks X_tt: {tt_ranks(X_tt)}, Z_tt: {tt_ranks(Z_tt)}, \n"
        f"      Y_tt: {tt_ranks(Y_tt)}, T_tt: {tt_ranks(T_tt) if T_tt else None} \n"
    )
    return X_tt, Y_tt, T_tt, Z_tt, {"num_iters": iteration}