import sys
import os

import numpy as np
from pygments.unistring import xid_start

sys.path.append(os.getcwd() + '/../')

from src.tt_ops import *
from src.tt_amen import tt_block_mals, _block_local_product, cached_einsum, tt_amen
from src.tt_ineq_check import tt_pd_optimal_step_size, tt_ineq_optimal_step_size
from dataclasses import dataclass

def forward_backward_sub(L, b, overwrite_b=False):
    y = scip.linalg.solve_triangular(L, b, lower=True, check_finite=False, overwrite_b=overwrite_b)
    x = scip.linalg.solve_triangular(L.T, y, lower=False, check_finite=False, overwrite_b=True)
    return x

def _ipm_block_local_product(XAX_k, block_A_k, XAX_kp1, x_core, inv_I):
    result = np.zeros_like(x_core)
    result[0] = (
            cached_einsum('lsr,smnS,LSR,rnR->lmL', XAX_k[(0, 0)], block_A_k[(0, 0)], XAX_kp1[(0, 0)], x_core[0]) # K y
            + cached_einsum('lsr,smnS,LSR,rnR->lmL', XAX_k[(0, 1)], block_A_k[(0, 1)], XAX_kp1[(0, 1)], x_core[1]) # -L x
    )
    result[1] = -inv_I*cached_einsum('lsr,smnS,LSR,lmL->rnR', XAX_k[(0, 1)], block_A_k[(0, 1)], XAX_kp1[(0, 1)], x_core[0]) # invI*L^* y
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
        try:
            L_L_Z = scip.linalg.cholesky(
                cached_einsum('lsr,smnS,LSR->lmLrnR', XAX_k[(2, 1)], block_A_k[(2, 1)], XAX_k1[(2, 1)]).reshape(m, m),
                check_finite=False, lower=True, overwrite_a=True
            )
            mR_p = rhs[:, 0].reshape(m, 1)
            mR_d = rhs[:, 1].reshape(m, 1)
            mR_c = rhs[:, 2].reshape(m, 1)
            L_X = cached_einsum('lsr,smnS,LSR->lmLrnR', XAX_k[(2, 2)], block_A_k[(2, 2)], XAX_k1[(2, 2)]).reshape(m, m)
            mL_eq = cached_einsum('lsr,smnS,LSR->lmLrnR', XAX_k[(0, 1)], block_A_k[(0, 1)], XAX_k1[(0, 1)]).reshape(m, m)
            A = mL_eq @ forward_backward_sub(L_L_Z, L_X * inv_I.reshape(1, -1), overwrite_b=True) @ mL_eq.T + cached_einsum('lsr,smnS,LSR->lmLrnR', XAX_k[(0, 0)], block_A_k[(0, 0)], XAX_k1[(0, 0)]).reshape(m, m)
            b = mR_p - mL_eq @ forward_backward_sub(L_L_Z, mR_c - (L_X * inv_I.reshape(1, -1)) @ mR_d, overwrite_b=True) - A @ np.transpose(previous_solution, (1, 0, 2, 3))[1].reshape(-1, 1)
            y = scip.linalg.solve(A, b, check_finite=False, overwrite_a=True, overwrite_b=True) + np.transpose(previous_solution, (1, 0, 2, 3))[1].reshape(-1, 1)
            z = inv_I.reshape(-1, 1) * (mR_d - mL_eq.T @ y)
            x = forward_backward_sub(L_L_Z, mR_c - L_X @ z, overwrite_b=True)
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
        solution_now, info = scip.sparse.linalg.bicgstab(
            linear_op,
            local_rhs.reshape(-1, 1),
            rtol=1e-3*block_res_old,
            maxiter=50
        )
        solution_now = np.transpose(solution_now.reshape(2, x_shape[0], x_shape[2], x_shape[3]), (1, 0, 2, 3)) + previous_solution[:, :2]
        z = inv_I * (rhs[:, 1] - cached_einsum('lsr,smnS,LSR,lmL->rnR', XAX_k[(0, 1)], block_A_k[(0, 1)], XAX_k1[(0, 1)], solution_now[:, 0]))
        solution_now = np.concatenate((solution_now, z.reshape(x_shape[0], 1, x_shape[2], x_shape[3])), axis=1)

    block_res_new = np.linalg.norm(_block_local_product(XAX_k, block_A_k, XAX_k1, solution_now).__isub__(rhs)) / norm_rhs

    if block_res_old < block_res_new:
        print("Triggered!!!!")
        solution_now = previous_solution

    return solution_now, block_res_old, min(block_res_new, block_res_old), rhs, norm_rhs


def _ipm_block_local_product_ineq(XAX_k, block_A_k, XAX_kp1, x_core, inv_I):
    result = np.zeros_like(x_core)
    result[0] = (
            cached_einsum('lsr,smnS,LSR,rnR->lmL', XAX_k[(0, 0)], block_A_k[(0, 0)], XAX_kp1[(0, 0)], x_core[0]) # K y
            + cached_einsum('lsr,smnS,LSR,rnR->lmL', XAX_k[(0, 1)], block_A_k[(0, 1)], XAX_kp1[(0, 1)], x_core[1]) # -L x
    )
    result[1] = -inv_I*cached_einsum('lsr,smnS,LSR,lmL->rnR', XAX_k[(0, 1)], block_A_k[(0, 1)], XAX_kp1[(0, 1)], x_core[0]) # invI*L^* y
    result[1] = cached_einsum('lsr,smnS,LSR,rnR->lmL', XAX_k[(2, 2)], block_A_k[(2, 2)], XAX_kp1[(2, 2)], result[1] - x_core[2]) # L_X
    result[1] += cached_einsum('lsr,smnS,LSR,rnR->lmL', XAX_k[(2, 1)], block_A_k[(2, 1)], XAX_kp1[(2, 1)], x_core[1]) # L_Z x
    result[2] = (
        cached_einsum('lsr,smnS,LSR,rnR->lmL', XAX_k[(3, 1)], block_A_k[(3, 1)], XAX_kp1[(3, 1)], x_core[1])
        + cached_einsum('lsr,smnS,LSR,rnR->lmL', XAX_k[(3, 3)], block_A_k[(3, 3)], XAX_kp1[(3, 3)], x_core[2])
    )
    return result

def _ipm_local_solver_ineq(XAX_k, block_A_k, XAX_k1, Xb_k, block_b_k, Xb_k1, previous_solution, nrmsc, size_limit, rtol):
    x_shape = previous_solution.shape
    m = x_shape[0] * x_shape[2] * x_shape[3]
    rhs = np.zeros_like(previous_solution)
    rhs[:, 0] = cached_einsum('br,bmB,BR->rmR', Xb_k[0], nrmsc * block_b_k[0], Xb_k1[0]) if 0 in block_b_k else 0
    rhs[:, 1] = cached_einsum('br,bmB,BR->rmR', Xb_k[1], nrmsc * block_b_k[1], Xb_k1[1]) if 1 in block_b_k else 0
    rhs[:, 2] = cached_einsum('br,bmB,BR->rmR', Xb_k[2], nrmsc * block_b_k[2], Xb_k1[2]) if 2 in block_b_k else 0
    rhs[:, 3] = cached_einsum('br,bmB,BR->rmR', Xb_k[3], nrmsc * block_b_k[3], Xb_k1[3]) if 3 in block_b_k else 0
    inv_I = np.divide(1, cached_einsum('lsr,smnS,LSR->lmL', XAX_k[(1, 2)], block_A_k[(1, 2)], XAX_k1[(1, 2)]))
    norm_rhs = np.linalg.norm(rhs)
    block_res_old_scalar = np.linalg.norm(_block_local_product(XAX_k, block_A_k, XAX_k1, previous_solution) - rhs) / norm_rhs
    if block_res_old_scalar < rtol:
        return previous_solution, block_res_old_scalar, block_res_old_scalar, rhs, norm_rhs

    size_limit = np.inf
    if m <= size_limit:
        try:
            L_L_Z = scip.linalg.cholesky(
                cached_einsum('lsr,smnS,LSR->lmLrnR', XAX_k[(2, 1)], block_A_k[(2, 1)], XAX_k1[(2, 1)]).reshape(m, m),
                check_finite=False, lower=True
            )
            mR_p = rhs[:, 0].reshape(m, 1)
            mR_d = rhs[:, 1].reshape(m, 1)
            L_L_Z_inv_mR_c = forward_backward_sub(L_L_Z, rhs[:, 2].reshape(m, 1))
            mR_t = rhs[:, 3].reshape(m, 1)
            L_X = cached_einsum('lsr,smnS,LSR->lmLrnR', XAX_k[(2, 2)], block_A_k[(2, 2)], XAX_k1[(2, 2)]).reshape(m, m)
            mL_eq = cached_einsum('lsr,smnS,LSR->lmLrnR', XAX_k[(0, 1)], block_A_k[(0, 1)], XAX_k1[(0, 1)]).reshape(m,m)
            T_op = cached_einsum('lsr,smnS,LSR->lmLrnR', XAX_k[(3, 1)], block_A_k[(3, 1)], XAX_k1[(3, 1)]).reshape(m, m)
            A = cached_einsum('lsr,smnS,LSR->lmLrnR', XAX_k[(0, 0)], block_A_k[(0, 0)],XAX_k1[(0, 0)]).reshape(m, m) + (mL_eq @ forward_backward_sub(L_L_Z, L_X * inv_I.reshape(1, -1) @ mL_eq.T))
            B = mL_eq @ forward_backward_sub(L_L_Z, L_X)
            C = T_op @ forward_backward_sub(L_L_Z, (L_X * inv_I.reshape(1, -1)) @ mL_eq.T)
            D = cached_einsum('lsr,smnS,LSR->lmLrnR', XAX_k[(3, 3)], block_A_k[(3, 3)], XAX_k1[(3, 3)]).reshape(m, m) + (T_op @ forward_backward_sub(L_L_Z, L_X))

            u = (
                    mR_p - mL_eq @ (L_L_Z_inv_mR_c - forward_backward_sub(L_L_Z, (L_X * inv_I.reshape(1, -1)) @ mR_d))
                    - A @ previous_solution[:, 1].reshape(-1, 1)
                    - B @ previous_solution[:, 3].reshape(-1, 1)
            )
            v = (
                    mR_t - T_op @ (L_L_Z_inv_mR_c - forward_backward_sub(L_L_Z, (L_X * inv_I.reshape(1, -1)) @ mR_d))
                    - C @ previous_solution[:, 1].reshape(-1, 1)
                    - D @ previous_solution[:, 3].reshape(-1, 1)
            )
            Dlu, Dpiv = scip.linalg.lu_factor(D, check_finite=False)
            lhs_l = A - B @ scip.linalg.lu_solve((Dlu, Dpiv), C, check_finite=False)
            rhs_l = u - B @ scip.linalg.lu_solve((Dlu, Dpiv), v, check_finite=False)
            y = scip.linalg.lu_solve(scip.linalg.lu_factor(lhs_l, check_finite=False), rhs_l, check_finite=False)
            t = scip.linalg.lu_solve((Dlu, Dpiv), v - C @ y, check_finite=False)
            y += previous_solution[:, 1].reshape(-1, 1)
            t += previous_solution[:, 3].reshape(-1, 1)
            z = inv_I.reshape(-1, 1) * (mR_d - mL_eq.T @ y) - t
            x = L_L_Z_inv_mR_c - forward_backward_sub(L_L_Z, L_X @ z)

            solution_now = np.transpose(
                np.vstack((y, x, z, t)).reshape(x_shape[1], x_shape[0], x_shape[2], x_shape[3]),
                (1, 0, 2, 3)
            )
        except:
            size_limit = 0

    if m > size_limit:
        def mat_vec(x_vec):
            x_vec = _ipm_block_local_product_ineq(
                XAX_k, block_A_k, XAX_k1,
                x_vec.reshape(3, x_shape[0], x_shape[2], x_shape[3]), inv_I
            ).reshape(-1, 1)
            return x_vec

        linear_op = scip.sparse.linalg.LinearOperator((3 * m, 3 * m), matvec=mat_vec)
        local_rhs = -_ipm_block_local_product_ineq(XAX_k, block_A_k, XAX_k1,
                                                   np.transpose(previous_solution[:, [0, 1, 3]], (1, 0, 2, 3)), inv_I)
        local_rhs[0] += rhs[:, 0]
        local_rhs[1] += rhs[:, 2] - cached_einsum('lsr,smnS,LSR,rnR->lmL', XAX_k[(2, 2)], block_A_k[(2, 2)],
                                                  XAX_k1[(2, 2)], inv_I * rhs[:, 1])
        local_rhs[2] += rhs[:, 3]
        solution_now, _ = scip.sparse.linalg.bicgstab(
            linear_op,
            local_rhs.reshape(-1, 1),
            rtol=1e-3 * block_res_old_scalar,
            maxiter=50
        )
        solution_now = np.transpose(solution_now.reshape(3, x_shape[0], x_shape[2], x_shape[3]),
                                    (1, 0, 2, 3)) + previous_solution[:, [0, 1, 3]]
        z = inv_I * (
                    rhs[:, 1] - cached_einsum('lsr,smnS,LSR,lmL->rnR', XAX_k[(0, 1)], block_A_k[(0, 1)], XAX_k1[(0, 1)],
                                              solution_now[:, 0])) - solution_now[:, 2]
        solution_now = np.concatenate(
            (solution_now[:, :2], z.reshape(x_shape[0], 1, x_shape[2], x_shape[3]), solution_now[:, None, -1]), axis=1)

    block_res_new_scalar = np.linalg.norm(_block_local_product(XAX_k, block_A_k, XAX_k1, solution_now) - rhs) / norm_rhs

    if block_res_old_scalar < block_res_new_scalar:
        solution_now = previous_solution

    return solution_now, block_res_old_scalar, min(block_res_new_scalar, block_res_old_scalar), rhs, norm_rhs

def tt_infeasible_newton_system(
        lhs,
        obj_tt,
        X_tt,
        Y_tt,
        Z_tt,
        T_tt,
        lin_op_tt,
        lin_op_tt_adj,
        bias_tt,
        ineq_mask,
        status
):
    rhs = {}
    if status.aho_direction:
        lhs[(2, 1)] = tt_rank_reduce(tt_scale(0.5, tt_add(tt_IkronM(Z_tt), tt_MkronI(Z_tt))), eps=status.op_tol, rank_weighted_error=True)
        lhs[(2, 2)] = tt_rank_reduce(tt_scale(0.5, tt_add(tt_MkronI(X_tt), tt_IkronM(X_tt))), eps=status.op_tol, rank_weighted_error=True)
    else:
        lhs[(2, 1)] = tt_rank_reduce(tt_MkronI(Z_tt), eps=status.op_tol, rank_weighted_error=True)
        lhs[(2, 2)] = tt_rank_reduce(tt_IkronM(X_tt), eps=status.op_tol, rank_weighted_error=True)

    # Check primal feasibility and compute residual
    primal_feas = tt_rank_reduce(tt_sub(tt_fast_matrix_vec_mul(lin_op_tt, tt_reshape(X_tt, (4, )), status.eps), bias_tt), status.op_tol, rank_weighted_error=True)  # primal feasibility
    status.primal_error = tt_norm(primal_feas) / status.primal_error_normalisation
    status.is_primal_feasible = status.primal_error < status.feasibility_tol

    # Check dual feasibility and compute residual
    dual_feas = tt_rank_reduce(tt_sub(tt_fast_matrix_vec_mul(lin_op_tt_adj, Y_tt, status.eps), tt_rank_reduce(tt_add(tt_reshape(Z_tt, (4, )), obj_tt), status.eps)), status.eps if status.with_ineq else status.op_tol, rank_weighted_error=not status.with_ineq)
    if status.with_ineq:
        dual_feas = tt_rank_reduce(tt_sub(dual_feas, tt_reshape(T_tt, (4, ))), status.op_tol, rank_weighted_error=True)
    status.dual_error = tt_norm(dual_feas) / status.dual_error_normalisation
    status.is_dual_feasible = status.dual_error < 2*status.feasibility_tol

    status.is_last_iter = status.is_last_iter or (status.is_primal_feasible and status.is_dual_feasible and status.is_central)

    if not status.is_primal_feasible or status.is_last_iter:
        rhs[0] = primal_feas

    if not status.is_dual_feasible or status.is_last_iter:
        rhs[1] = dual_feas

    if not status.is_central or status.is_last_iter:
        if status.aho_direction:
            rhs[2] = tt_reshape(tt_scale(-1, _tt_symmetrise(tt_fast_mat_mat_mul(X_tt, Z_tt, status.eps), status.op_tol)), (4, ))
        else:
            rhs[2] = tt_reshape(tt_scale(-1, tt_rank_reduce(tt_fast_mat_mat_mul(Z_tt, X_tt, status.eps), eps=status.op_tol, rank_weighted_error=True)), (4,))

    if status.with_ineq:
        # TODO: There might be room to optimise the tt_diag(tt_split_bonds(.))-expression
        lhs[(3, 1)] =  tt_diag_op(T_tt, status.op_tol, rank_weighted_error=True)
        masked_X_tt = tt_rank_reduce(tt_add(tt_scale(status.boundary_val, ineq_mask), tt_fast_hadammard(ineq_mask, X_tt, status.eps)), eps=status.eps)
        lhs[(3, 3)] = tt_rank_reduce(tt_add(status.lag_map_t, tt_diag_op(masked_X_tt, status.eps)), eps=status.op_tol, rank_weighted_error=True)
        if not status.is_central or status.is_last_iter:
            rhs[3] = tt_reshape(tt_rank_reduce(tt_scale(-1, tt_fast_hadammard(masked_X_tt, T_tt, status.eps)), eps=status.op_tol, rank_weighted_error=True), (4, ))

    return lhs, rhs, status

def _tt_symmetrise(matrix_tt, err_bound):
    return tt_rank_reduce(tt_scale(0.5, tt_add(matrix_tt, tt_transpose(matrix_tt))), eps=err_bound, rank_weighted_error=True)

def _tt_get_block(i, block_matrix_tt):
    if len(block_matrix_tt[0].shape) < len(block_matrix_tt[-1].shape):
        return block_matrix_tt[:-1] + [block_matrix_tt[-1][:, i]]
    return [block_matrix_tt[0][:, i]] + block_matrix_tt[1:]

def _tt_ipm_newton_step(
        lhs_matrix_tt,
        rhs_vec_tt,
        ineq_mask,
        X_tt,
        Z_tt,
        T_tt,
        ZX,
        TX,
        status,
        solver
):

    # Predictor
    if status.verbose:
        print("--- Predictor  step ---")
    Delta_tt, res = solver(lhs_matrix_tt, rhs_vec_tt, None, 8 + status.is_last_iter*2 + status.with_ineq)
    Delta_X_tt = _tt_symmetrise(tt_reshape(_tt_get_block(1, Delta_tt), (2, 2)), status.op_tol)
    Delta_Z_tt = _tt_symmetrise(tt_reshape(_tt_get_block(2, Delta_tt), (2, 2)), status.op_tol)
    Delta_Y_tt = tt_rank_reduce(_tt_get_block(0, Delta_tt), eps=status.op_tol, rank_weighted_error=True)
    Delta_Y_tt = tt_rank_reduce(tt_sub(Delta_Y_tt, tt_fast_matrix_vec_mul(status.lag_map_y, Delta_Y_tt, status.eps)), eps=status.op_tol, rank_weighted_error=True)
    Delta_T_tt = _tt_symmetrise(tt_fast_hadammard(ineq_mask, tt_reshape(_tt_get_block(3, Delta_tt), (2, 2)), status.eps), 0.1*status.op_tol) if status.with_ineq else None

    x_step_size, z_step_size, permitted_error = _tt_line_search(
        X_tt,
        Z_tt,
        T_tt,
        Delta_X_tt,
        Delta_Z_tt,
        Delta_T_tt,
        ineq_mask,
        status
    )

    if not status.is_central:
        # Corrector
        if status.verbose:
            print("\n--- Centering-Corrector  step ---")

        if status.aho_direction:
            Delta_XZ_term = tt_scale(-1, _tt_symmetrise(tt_fast_mat_mat_mul(Delta_X_tt, Delta_Z_tt, status.eps), status.op_tol))
        else:
            Delta_XZ_term = tt_scale(-1, tt_fast_mat_mat_mul(Delta_Z_tt, Delta_X_tt, status.op_tol))

        if status.with_ineq:
            mu_aff = (
                ZX + x_step_size * z_step_size * tt_inner_prod(Delta_X_tt, Delta_Z_tt)
                + z_step_size * tt_inner_prod(X_tt, Delta_Z_tt)
                + x_step_size * tt_inner_prod(Delta_X_tt, Z_tt)
                + TX + x_step_size * z_step_size * tt_inner_prod(Delta_T_tt, Delta_X_tt)
                + z_step_size * (tt_inner_prod(X_tt, Delta_T_tt) + status.boundary_val*tt_entrywise_sum(Delta_T_tt))
                + x_step_size * tt_inner_prod(Delta_X_tt, T_tt)
            )
            status.sigma = (mu_aff/(ZX + TX))**3
            rhs_3 = tt_add(
                    tt_scale(status.sigma * status.mu, tt_reshape(ineq_mask, (4,))),
                    tt_sub(
                        rhs_vec_tt[3],
                        tt_reshape(tt_fast_hadammard(Delta_T_tt, Delta_X_tt, status.op_tol), (4,))
                    )
                ) if status.sigma > 0 else tt_sub(rhs_vec_tt[3], tt_reshape(tt_fast_hadammard(Delta_T_tt, Delta_X_tt, status.op_tol), (4,)))
            rhs_vec_tt[3] = tt_rank_reduce(
                rhs_3,
                status.op_tol,
                rank_weighted_error=True
        )
        else:
            mu_aff = (
                ZX + x_step_size * z_step_size * tt_inner_prod(Delta_X_tt, Delta_Z_tt)
                + z_step_size * tt_inner_prod(X_tt,Delta_Z_tt)
                + x_step_size * tt_inner_prod(Delta_X_tt, Z_tt)
            )
            status.sigma = (mu_aff/ZX) ** 3


        rhs_vec_tt[2] = tt_rank_reduce(
            tt_add(
                tt_scale(status.sigma*status.mu, tt_reshape(tt_identity(len(X_tt)), (4, ))),
                tt_add(
                rhs_vec_tt[2],
                tt_reshape(Delta_XZ_term, (4, ))
                )
            ),
            status.op_tol,
            rank_weighted_error=True
        ) if status.sigma > 0 else tt_add(rhs_vec_tt[2], tt_reshape(Delta_XZ_term, (4, )))

        Delta_tt_cc, res = solver(lhs_matrix_tt, rhs_vec_tt, Delta_tt, 6 + status.with_ineq)
        Delta_X_tt_cc = _tt_symmetrise(tt_reshape(_tt_get_block(1, Delta_tt_cc), (2, 2)), status.op_tol)
        Delta_Z_tt_cc = _tt_symmetrise(tt_reshape(_tt_get_block(2, Delta_tt_cc), (2, 2)), status.op_tol)
        Delta_X_tt = tt_rank_reduce(tt_add(Delta_X_tt_cc, Delta_X_tt), eps=status.op_tol)
        Delta_Z_tt = tt_rank_reduce(tt_add(Delta_Z_tt_cc, Delta_Z_tt), eps=status.op_tol)
        if status.with_ineq:
            Delta_T_tt_cc = _tt_symmetrise(tt_fast_hadammard(ineq_mask, tt_reshape(_tt_get_block(3, Delta_tt_cc), (2, 2)), status.eps), 0.1*status.op_tol)
            Delta_T_tt = tt_rank_reduce(tt_add(Delta_T_tt_cc, Delta_T_tt), eps=0.1*status.op_tol, rank_weighted_error=True)

        x_step_size, z_step_size, permitted_error = _tt_line_search(
            X_tt,
            Z_tt,
            T_tt,
            Delta_X_tt,
            Delta_Z_tt,
            Delta_T_tt,
            ineq_mask,
            status
        )
        Delta_Y_tt_cc = tt_rank_reduce(_tt_get_block(0, Delta_tt_cc), eps=status.eps)
        Delta_Y_tt_cc = tt_rank_reduce(tt_sub(Delta_Y_tt_cc, tt_fast_matrix_vec_mul(status.lag_map_y, Delta_Y_tt_cc, status.eps)), eps=status.eps, rank_weighted_error=True)
        Delta_Y_tt = tt_rank_reduce(tt_add(Delta_Y_tt_cc, Delta_Y_tt), eps=status.op_tol, rank_weighted_error=True)
    else:
        status.sigma = 0

    return x_step_size, z_step_size, Delta_X_tt, Delta_Y_tt, Delta_Z_tt, Delta_T_tt, status, permitted_error


def _tt_line_search(
        X_tt,
        Z_tt,
        T_tt,
        Delta_X_tt,
        Delta_Z_tt,
        Delta_T_tt,
        ineq_mask,
        status
):
    if status.is_last_iter:
        x_step_size = 1
        z_step_size = 1
        permitted_err_x = 1
        permitted_err_z = 1
    else:
        x_step_size, permitted_err_x = tt_pd_optimal_step_size(X_tt, Delta_X_tt, status.op_tol, tol=status.eps)
        z_step_size, permitted_err_z = tt_pd_optimal_step_size(Z_tt, Delta_Z_tt, status.op_tol, tol=status.eps)
    if status.with_ineq and not status.is_last_iter:
        x_step_size, z_step_size, permitted_err_ineq = _tt_line_search_ineq(x_step_size, z_step_size, X_tt, T_tt, Delta_X_tt, Delta_T_tt, ineq_mask, status)
        permitted_err_x = min(permitted_err_x, permitted_err_ineq[0])
        permitted_err_z = min(permitted_err_z, permitted_err_ineq[1])
    tau_x = 0.95 + 0.05*(permitted_err_x >= status.op_tol or x_step_size == 1)
    tau_z = 0.95 + 0.05*(permitted_err_z >= status.op_tol or z_step_size == 1)
    return tau_x*x_step_size, tau_z*z_step_size, (permitted_err_x, permitted_err_z)


def _tt_line_search_ineq(x_step_size, z_step_size, X_tt, T_tt, Delta_X_tt, Delta_T_tt, ineq_mask, status):
    if x_step_size > 0:
        degeneracy_helper = tt_scale(status.boundary_val, tt_one_matrix(len(X_tt)))
        masked_X_tt = tt_rank_reduce(tt_add(tt_fast_hadammard(ineq_mask, X_tt, status.eps), tt_scale(0.9, degeneracy_helper)), status.op_tol, rank_weighted_error=True)
        masked_Delta_X_tt = tt_rank_reduce(
            tt_add(
                tt_scale(x_step_size, tt_fast_hadammard(ineq_mask, Delta_X_tt, status.eps)),
                tt_scale(0.1, degeneracy_helper)
            ),
            status.op_tol,
            rank_weighted_error=True
        )
        x_ineq_step_size, permitted_error_x = tt_ineq_optimal_step_size(
            tt_diag_op(masked_X_tt, status.eps),
            tt_diag_op(masked_Delta_X_tt, status.eps),
            status.op_tol, verbose=status.verbose
        )
        x_step_size *= x_ineq_step_size

    if z_step_size > 0:
        # FIXME
        # TODO: T_tt becomes negative
        # FIXME
        print()
        print("Panic!!!!!!!!!", np.min(tt_matrix_to_matrix(T_tt)))
        masked_T_tt = tt_rank_reduce(tt_add(T_tt, status.compl_ineq_mask), status.op_tol, rank_weighted_error=True)
        masked_Delta_T_tt = tt_rank_reduce(
            tt_add(tt_scale(z_step_size, Delta_T_tt), status.compl_ineq_mask),
            status.op_tol,
            rank_weighted_error=True
        )
        t_step_size, permitted_error_t = tt_ineq_optimal_step_size(
            tt_diag_op(masked_T_tt, status.op_tol),
            tt_diag_op(masked_Delta_T_tt, status.op_tol),
            status.op_tol, verbose=status.verbose
        )
        z_step_size *= t_step_size

    return x_step_size, z_step_size, (permitted_error_x, permitted_error_t)


def _update(x_step_size, z_step_size, X_tt, Z_tt, Delta_X_tt, Delta_Z_tt, op_tol, permitted_error):
    if permitted_error[0] <= 0:
        X_tt = tt_rank_reduce(tt_add(X_tt, tt_scale(op_tol, tt_identity(len(X_tt)))), eps=op_tol, rank_weighted_error=True)
    else:
        X_tt = tt_rank_reduce(tt_add(X_tt, tt_scale(x_step_size, Delta_X_tt)), eps=op_tol, rank_weighted_error=True)
    if permitted_error[1] <= 0:
        Z_tt = tt_rank_reduce(tt_add(Z_tt, tt_scale(op_tol, tt_identity(len(Z_tt)))), eps=op_tol, rank_weighted_error=True)
    else:
        Z_tt = tt_rank_reduce(tt_add(Z_tt, tt_scale(z_step_size, Delta_Z_tt)), eps=op_tol, rank_weighted_error=True)
    return X_tt, Z_tt

def _initialise(obj_tt, lin_op_tt, bias_tt, ineq_mask, status, dim):
    lin_op_tt_T = tt_transpose(lin_op_tt)
    A = tt_rank_reduce(tt_add(tt_fast_mat_mat_mul(lin_op_tt, lin_op_tt_T, status.eps), tt_fast_mat_mat_mul(tt_transpose(status.lag_map_y), status.lag_map_y, status.eps)), eps=status.op_tol, rank_weighted_error=True)
    Y_tt, _ = tt_amen(A, obj_tt)
    Y_tt = tt_rank_reduce(tt_sub(Y_tt, tt_fast_matrix_vec_mul(status.lag_map_y, Y_tt, status.eps)), eps=status.op_tol, rank_weighted_error=True)
    res_tt = tt_reshape(tt_rank_reduce(tt_sub(obj_tt, tt_fast_matrix_vec_mul(lin_op_tt, Y_tt, status.eps)), eps=status.op_tol, rank_weighted_error=True), (2, 2))
    T_tt = None
    if status.with_ineq:
        res_tt = tt_rank_reduce(tt_sub(res_tt, ineq_mask), eps=status.op_tol, rank_weighted_error=True)
        masked_res_tt = tt_fast_hadammard(ineq_mask, res_tt, status.eps)
        masked_res_tt_prime = tt_rank_reduce(tt_add(masked_res_tt, status.compl_ineq_mask), eps=status.op_tol,
                                             rank_weighted_error=True)
        t_step_size, permitted_err_t = tt_ineq_optimal_step_size(
            [np.eye(4).reshape(1, 4, 4, 1) for _ in range(dim)],
            tt_diag_op(masked_res_tt_prime, status.op_tol),
            status.op_tol, verbose=status.verbose
        )
        tau_t = 0.95
        T_tt = tt_rank_reduce(tt_add(ineq_mask, tt_scale(tau_t * t_step_size, masked_res_tt)), eps=0.1 * status.op_tol,
                              rank_weighted_error=True)
        res_tt = tt_rank_reduce(tt_sub(res_tt, T_tt), eps=status.op_tol, rank_weighted_error=True)
        #print(tt_matrix_to_matrix(T_tt))

    Z_tt = tt_identity(dim)
    res_tt = tt_rank_reduce(tt_sub(res_tt, Z_tt), eps=status.op_tol, rank_weighted_error=True)
    z_step_size, permitted_err_z = tt_pd_optimal_step_size(Z_tt, res_tt, status.op_tol, tol=status.eps)
    tau_z = 0.95 + 0.05 * (permitted_err_z >= status.op_tol or z_step_size == 1)
    Z_tt = tt_rank_reduce(tt_add(Z_tt, tt_scale(tau_z*z_step_size, res_tt)), eps=status.op_tol, rank_weighted_error=True)
    x_initial_step = tt_norm(bias_tt) / tt_norm(tt_diagonal(lin_op_tt))
    X_tt = tt_scale(x_initial_step, tt_identity(dim))
    #print(tt_matrix_to_matrix(X_tt))
    #print(tt_matrix_to_matrix(tt_reshape(Y_tt, (2, 2))))
    #print(tt_matrix_to_matrix(Z_tt))

    return X_tt, Y_tt, Z_tt, T_tt



@dataclass
class IPMStatus:
    feasibility_tol: float
    centrality_tol: float
    op_tol: float
    eps: float

    aho_direction: bool
    is_primal_feasible: bool
    primal_error: float
    is_dual_feasible: bool
    dual_error: float
    is_central: bool
    mu: float
    is_last_iter: bool
    with_ineq: bool
    verbose: bool

    primal_error_normalisation: float
    dual_error_normalisation: float

    boundary_val: float = 0.01
    sigma: float = 0.5
    num_ineq_constraints: float = 0
    lag_map_t: list = None
    lag_map_y: list = None
    compl_ineq_mask = None


def tt_ipm(
    lag_maps,
    obj_tt,
    lin_op_tt,
    bias_tt,
    ineq_mask=None,
    max_iter=100,
    feasibility_tol=1e-5,
    centrality_tol=1e-3,
    aho_direction=True,
    op_tol=1e-5,
    eps=1e-12,
    verbose=False,
):
    dim = len(obj_tt)
    status = IPMStatus(
        feasibility_tol,
        centrality_tol,
        op_tol,
        eps,
        aho_direction,
        False,
        np.inf,
        False,
        np.inf,
        False,
        np.inf,
        False,
        ineq_mask is not None,
        verbose,
        1,
        1
    )
    lhs_skeleton = {}
    if status.with_ineq:
        solver = lambda lhs, rhs, x0, nwsp: tt_block_mals(
            lhs,
            rhs,
            x0=x0,
            local_solver=_ipm_local_solver_ineq,
            tol=0.1 * min(feasibility_tol, centrality_tol),
            nswp=nwsp,
            verbose=verbose
        )
        status.num_ineq_constraints = tt_inner_prod(ineq_mask, ineq_mask)
        status.compl_ineq_mask = tt_rank_reduce(tt_sub(tt_one_matrix(dim), ineq_mask), eps=op_tol, rank_weighted_error=True)
        status.lag_map_t = lag_maps["t"]
        lhs_skeleton[(1, 3)] = tt_reshape(tt_identity(2 * dim), (4, 4))
    else:
        solver = lambda lhs, rhs, x0, nwsp: tt_block_mals(
            lhs,
            rhs,
            x0=x0,
            local_solver=_ipm_local_solver,
            tol=0.1 * min(feasibility_tol, centrality_tol),
            nswp=nwsp,
            verbose=verbose
        )
        status.num_ineq_constraints = 0

    lag_maps = {key: tt_rank_reduce(value, eps=op_tol) for key, value in lag_maps.items()}
    obj_tt = tt_rank_reduce(obj_tt, eps=op_tol)
    lin_op_tt = tt_rank_reduce(lin_op_tt, eps=op_tol)
    bias_tt = tt_rank_reduce(bias_tt, eps=op_tol)

    # Normalisation
    # We normalise the objective to the scale of the average constraint
    status.primal_error_normalisation = 1 + tt_norm(bias_tt)
    status.dual_error_normalisation = 1 + tt_norm(obj_tt)

    # KKT-system prep
    lin_op_tt_adj = tt_transpose(lin_op_tt)
    lhs_skeleton[(0, 1)] = tt_scale(-1, lin_op_tt)
    lhs_skeleton[(1, 0)] = tt_scale(-1, lin_op_tt_adj)
    lhs_skeleton[(0, 0)] = lag_maps["y"]
    status.lag_map_y = lag_maps["y"]
    lhs_skeleton[(1, 2)] = tt_reshape(tt_identity(2 * dim), (4, 4))

    X_tt, Y_tt, Z_tt, T_tt = _initialise(obj_tt, lin_op_tt, bias_tt, ineq_mask, status, dim)

    iteration = 0
    finishing_steps = 1
    while finishing_steps > 0:
        #print()
        #print("Norms: ", tt_norm(X_tt), tt_norm(Y_tt), tt_norm(Z_tt), tt_norm(T_tt), np.min(tt_matrix_to_matrix(T_tt)))
        #print()
        iteration += 1
        ZX = tt_inner_prod(Z_tt, X_tt)
        TX = tt_inner_prod(X_tt, T_tt) + status.boundary_val*tt_entrywise_sum(T_tt) if status.with_ineq else 0
        status.mu = np.divide(ZX + TX, (2 ** dim + status.num_ineq_constraints))
        status.is_central = np.less(status.mu / (1 + tt_inner_prod(obj_tt, tt_reshape(X_tt, (4, )))), (1 + status.with_ineq)*centrality_tol)
        status.is_last_iter = status.is_last_iter or (max_iter == iteration)

        lhs_matrix_tt, rhs_vec_tt, status = tt_infeasible_newton_system(
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
            status
        )


        if verbose:
            print(f"\nResults of iteration {iteration-1}:")
            print(f"Finishing up: {status.is_last_iter}")
            print(f"Is central: {status.is_central}, Is primal feasible:  {status.is_primal_feasible}, Is dual feasible: {status.is_dual_feasible}")
            print(f"Using AHO-Direction: {status.aho_direction}")
            print(f"Avg Compl. Slackness: {status.mu}")
            print(f"Primal-Dual error: {status.primal_error, status.dual_error}")
            print(f"Sigma: {status.sigma}")
            print(
                f"Ranks X_tt: {tt_ranks(X_tt)}, Z_tt: {tt_ranks(Z_tt)}, \n"
                f"      Y_tt: {tt_ranks(Y_tt)}, T_tt: {tt_ranks(T_tt) if T_tt else None} \n"
            )
            print(f"--- Step {iteration} ---")

        x_step_size, z_step_size, Delta_X_tt, Delta_Y_tt, Delta_Z_tt, Delta_T_tt, status, permitted_error = _tt_ipm_newton_step(
            lhs_matrix_tt,
            rhs_vec_tt,
            ineq_mask,
            X_tt,
            Z_tt,
            T_tt,
            ZX,
            TX,
            status,
            solver
        )
        if verbose:
            print(f"Step sizes: {x_step_size:.4f}, {z_step_size:.4f}")

        X_tt, Z_tt = _update(x_step_size, z_step_size, X_tt, Z_tt, Delta_X_tt, Delta_Z_tt, op_tol, permitted_error)
        Y_tt = tt_rank_reduce(tt_add(Y_tt, tt_scale(z_step_size, Delta_Y_tt)), eps=op_tol, rank_weighted_error=True)

        if status.with_ineq:
            T_tt = tt_rank_reduce(tt_add(T_tt, tt_scale(z_step_size, Delta_T_tt)), eps=0.1*op_tol, rank_weighted_error=True)
            #T_tt = tt_rank_reduce(tt_add(T_tt, tt_scale(op_tol, ineq_mask)), eps=0.1*op_tol, rank_weighted_error=True)
        if status.is_last_iter:
            finishing_steps -= 1

    print(f"---Terminated---")
    print(f"Converged in {iteration} iterations.")
    print(
        f"Ranks X_tt: {tt_ranks(X_tt)}, Z_tt: {tt_ranks(Z_tt)}, \n"
        f"      Y_tt: {tt_ranks(Y_tt)}, T_tt: {tt_ranks(T_tt) if T_tt else None} \n"
    )
    return X_tt, Y_tt, T_tt, Z_tt, {"num_iters": iteration}