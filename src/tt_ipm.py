import sys
import os
import numpy as np
import traceback

sys.path.append(os.getcwd() + '/../')

from src.tt_ops import *
from src.tt_als import cached_einsum, TTBlockMatrix, TTBlockVector, tt_max_generalised_eigen, tt_min_eig, tt_mat_mat_mul,tt_restarted_block_amen
from dataclasses import dataclass
from enum import Enum
import warnings
warnings.simplefilter("error")

class IneqStatus(Enum):
    """
    Represents the status of an inequality constraint with specific integer values.
    """
    ACTIVE = 0           # Constraint is active (e.g., g(x) = 0)
    SETTING_ACTIVE = 1   # Constraint is in the process of becoming active
    SETTING_INACTIVE = 2 # Constraint is in the process of becoming inactive
    INACTIVE = 3         # Constraint is inactive (e.g., g(x) < 0)
    NOT_IN_USE = 4

    def __str__(self):
        return self.name.lower().replace('_', ' ')

def forward_backward_sub(L, b, overwrite_b=False):
    y = scp.linalg.solve_triangular(L, b, lower=True, check_finite=False, overwrite_b=overwrite_b)
    x = scp.linalg.solve_triangular(L.T, y, lower=False, check_finite=False, overwrite_b=True)
    return x

def _ipm_local_solver(XAX_k, block_A_k, XAX_k1, Xb_k, block_b_k, Xb_k1, previous_solution, size_limit, dense_solve=True, rtol=1e-5):
    x_shape = previous_solution.shape
    m = x_shape[0] * x_shape[2] * x_shape[3]
    rhs = np.empty_like(previous_solution)
    rhs[:, 0] = cached_einsum('br,bmB,BR->rmR', Xb_k[0], block_b_k[0], Xb_k1[0]) if 0 in block_b_k else 0
    rhs[:, 1] = cached_einsum('br,bmB,BR->rmR', Xb_k[1], block_b_k[1], Xb_k1[1]) if 1 in block_b_k else 0
    rhs[:, 2] = cached_einsum('br,bmB,BR->rmR', Xb_k[2], block_b_k[2], Xb_k1[2]) if 2 in block_b_k else 0
    norm_rhs = max(np.linalg.norm(rhs), 1e-10)
    inv_I = np.divide(1, cached_einsum('lsr,smnS,LSR->lmL', XAX_k[1, 2], block_A_k[1, 2], XAX_k1[1, 2]))
    block_res_old = np.linalg.norm(block_A_k.block_local_product(XAX_k, XAX_k1, previous_solution).__isub__(rhs)) / norm_rhs
    direct_solve_failure = not dense_solve
    dense_solve = (np.sqrt(x_shape[0]*x_shape[3]) <= size_limit) and dense_solve

    if block_res_old < rtol:
        return previous_solution, block_res_old, block_res_old, rhs, norm_rhs, direct_solve_failure
    
    if dense_solve:
        try:
            mR_p = rhs[:, 0].reshape(m, 1)
            mR_d = rhs[:, 1].reshape(m, 1)
            mR_c = rhs[:, 2].reshape(m, 1)
            L_X_I_inv = cached_einsum('lsr,smnS,LSR->lmLrnR', XAX_k[2, 2], block_A_k[2, 2], XAX_k1[2, 2]).reshape(m, m)
            L_X_I_inv *= inv_I.reshape(1, -1)
            mL_eq = cached_einsum('lsr,smnS,LSR->lmLrnR', XAX_k[0, 1], block_A_k[0, 1], XAX_k1[0, 1]).reshape(m, m)
            L_L_Z = scp.linalg.cholesky(
                cached_einsum('lsr,smnS,LSR->lmLrnR', XAX_k[2, 1], block_A_k[2, 1], XAX_k1[2, 1]).reshape(m, m),
                check_finite=False, lower=True, overwrite_a=True
            )
            b = mR_p - mL_eq @ forward_backward_sub(L_L_Z, mR_c - L_X_I_inv @ mR_d, overwrite_b=True)
            A = forward_backward_sub(L_L_Z, L_X_I_inv, overwrite_b=True)
            np.matmul(A, mL_eq.T, out=A)
            np.matmul(mL_eq, A, out=A)
            A += cached_einsum('lsr,smnS,LSR->lmLrnR',XAX_k[0, 0], block_A_k[0, 0], XAX_k1[0, 0]).reshape(m, m)
            A.flat[::A.shape[1] + 1] += 1e-11
            solution_now = np.empty(x_shape)
            solution_now[:, 0] = scp.linalg.solve(A, b, check_finite=False, overwrite_a=True, overwrite_b=True, assume_a="gen").reshape(x_shape[0], x_shape[2], x_shape[3])
            solution_now[:, 2] = (
                mR_d - cached_einsum('lsr,smnS,LSR,lmL->rnR', XAX_k[0, 1], block_A_k[0, 1], XAX_k1[0, 1], solution_now[:, 0]).reshape(-1, 1)
                ).__imul__(inv_I.reshape(-1, 1)).reshape(x_shape[0], x_shape[2], x_shape[3])
            solution_now[:, 1] = forward_backward_sub(
                L_L_Z, 
                mR_c - cached_einsum('lsr,smnS,LSR,rnR->lmL', XAX_k[2, 2], block_A_k[2, 2], XAX_k1[2, 2], solution_now[:, 2]).reshape(-1, 1), 
                overwrite_b=True
                ).reshape(x_shape[0], x_shape[2], x_shape[3])
        except Exception as e:
            tb = traceback.extract_tb(e.__traceback__)
            last = tb[-1]
            print(f"\t⚠️ {type(e).__name__} in {last.filename}, \n\tline {last.lineno}: {last.line.strip()}")
            direct_solve_failure = True

    if not dense_solve or direct_solve_failure:
        matvec_wrapper = MatVecWrapper(
            XAX_k[0, 0], XAX_k[0, 1], XAX_k[2, 1], XAX_k[2, 2],
            block_A_k[0, 0], block_A_k[0, 1], block_A_k[2, 1], block_A_k[2, 2],
            XAX_k1[0, 0], XAX_k1[0, 1], XAX_k1[2, 1], XAX_k1[2, 2],
            inv_I, x_shape[0], x_shape[2], x_shape[3]
        )
        op = scp.sparse.linalg.LinearOperator(
            shape=(2 * m, 2 * m),
            matvec=matvec_wrapper.matvec,
            dtype=np.float64
        )
        local_rhs = np.empty((2, x_shape[0], x_shape[2], x_shape[3]))
        local_rhs[0] = rhs[:, 0]
        local_rhs[1] = rhs[:, 2]
        local_rhs[1] -= cached_einsum('lsr,smnS,LSR,rnR->lmL', XAX_k[2, 2], block_A_k[2, 2], XAX_k1[2, 2], inv_I*rhs[:, 1])
        local_rhs_norm = np.linalg.norm(local_rhs)
        local_vec = op(np.transpose(previous_solution[:, :2], (1, 0, 2, 3)).flatten()).reshape(2, x_shape[0], x_shape[2], x_shape[3])
        local_rhs_norm_prime = np.linalg.norm(local_rhs - local_vec)
        use_prev_sol = (local_rhs_norm_prime < local_rhs_norm)
        if use_prev_sol:
            local_rhs -= local_vec

        solution_now, _ = scp.sparse.linalg.gcrotmk(op, local_rhs.flatten(), rtol=rtol, k=8, maxiter=50, m=25, truncate="smallest")
        solution_now = np.transpose(solution_now.reshape(2, x_shape[0], x_shape[2], x_shape[3]), (1, 0, 2, 3))

        if use_prev_sol:
            solution_now[:, :2] += previous_solution[:, :2]
            
        z = inv_I * (rhs[:, 1] - cached_einsum('lsr,smnS,LSR,lmL->rnR', XAX_k[0, 1], block_A_k[0, 1], XAX_k1[0, 1], solution_now[:, 0]))
        solution_now = np.concatenate((solution_now, z.reshape(x_shape[0], 1, x_shape[2], x_shape[3])), axis=1)

    block_res_new = np.linalg.norm(block_A_k.block_local_product(XAX_k, XAX_k1, solution_now).__isub__(rhs)) / norm_rhs

    if block_res_old < block_res_new:
        solution_now = previous_solution

    return solution_now, block_res_old, min(block_res_old, block_res_new), rhs, norm_rhs, direct_solve_failure

def _ipm_local_solver_ineq(XAX_k, block_A_k, XAX_k1, Xb_k, block_b_k, Xb_k1, previous_solution, size_limit, dense_solve=True, rtol=1e-5):
    x_shape = previous_solution.shape
    m = x_shape[0] * x_shape[2] * x_shape[3]
    rhs = np.empty_like(previous_solution)
    rhs[:, 0] = cached_einsum('br,bmB,BR->rmR', Xb_k[0], block_b_k[0], Xb_k1[0]) if 0 in block_b_k else 0
    rhs[:, 1] = cached_einsum('br,bmB,BR->rmR', Xb_k[1], block_b_k[1], Xb_k1[1]) if 1 in block_b_k else 0
    rhs[:, 2] = cached_einsum('br,bmB,BR->rmR', Xb_k[2], block_b_k[2], Xb_k1[2]) if 2 in block_b_k else 0
    rhs[:, 3] = cached_einsum('br,bmB,BR->rmR', Xb_k[3], block_b_k[3], Xb_k1[3]) if 3 in block_b_k else 0
    inv_I = np.divide(1, cached_einsum('lsr,smnS,LSR->lmL', XAX_k[1, 2], block_A_k[1, 2], XAX_k1[1, 2]))
    norm_rhs = max(np.linalg.norm(rhs), 1e-10)
    block_res_old = np.linalg.norm(block_A_k.block_local_product(XAX_k, XAX_k1, previous_solution).__isub__(rhs)) / norm_rhs
    direct_solve_failure = not dense_solve
    dense_solve = (np.sqrt(x_shape[0]*x_shape[3]) <= size_limit) and dense_solve

    if block_res_old < rtol:
        return previous_solution, block_res_old, block_res_old, rhs, norm_rhs, direct_solve_failure
            
    if dense_solve:
        try:
            L_L_Z = scp.linalg.cholesky(
                cached_einsum('lsr,smnS,LSR->lmLrnR', XAX_k[2, 1], block_A_k[2, 1], XAX_k1[2, 1]).reshape(m, m),
                check_finite=False, lower=True,  overwrite_a=True
            )
            mR_p = rhs[:, 0].reshape(m, 1)
            mR_d = rhs[:, 1].reshape(m, 1)
            mR_c = rhs[:, 2].reshape(m, 1)
            mR_t = rhs[:, 3].reshape(m, 1)
            L_L_Z_inv_mR_c = forward_backward_sub(L_L_Z, rhs[:, 2].reshape(m, 1))
            L_L_Z_inv_L_X = forward_backward_sub(L_L_Z, cached_einsum('lsr,smnS,LSR->lmLrnR', XAX_k[2, 2], block_A_k[2, 2], XAX_k1[2, 2]).reshape(m, m), overwrite_b=True)
            mL_eq = cached_einsum('lsr,smnS,LSR->lmLrnR', XAX_k[0, 1], block_A_k[0, 1], XAX_k1[0, 1]).reshape(m, m)
            T_op = cached_einsum('lsr,smnS,LSR->lmLrnR', XAX_k[3, 1], block_A_k[3, 1], XAX_k1[3, 1]).reshape(m, m)
            u = mR_p - mL_eq @ (L_L_Z_inv_mR_c - (L_L_Z_inv_L_X * inv_I.reshape(1, -1)) @ mR_d)
            v = mR_t - T_op @ (L_L_Z_inv_mR_c - (L_L_Z_inv_L_X * inv_I.reshape(1, -1)) @ mR_d)
            A = cached_einsum('lsr,smnS,LSR->lmLrnR', XAX_k[0, 0], block_A_k[0, 0],XAX_k1[0, 0]).reshape(m, m).__iadd__(mL_eq @ (L_L_Z_inv_L_X * inv_I.reshape(1, -1)) @ mL_eq.T)
            B = mL_eq @ L_L_Z_inv_L_X
            C = T_op @ (L_L_Z_inv_L_X * inv_I.reshape(1, -1)) @ mL_eq.T
            D = cached_einsum('lsr,smnS,LSR->lmLrnR', XAX_k[3, 3], block_A_k[3, 3], XAX_k1[3, 3]).reshape(m, m).__iadd__(T_op @ L_L_Z_inv_L_X)
            D.flat[::D.shape[1] + 1] += 1e-11
            Dlu, Dpiv = scp.linalg.lu_factor(D, check_finite=False, overwrite_a=True)
            rhs_l = u.__isub__(B @ scp.linalg.lu_solve((Dlu, Dpiv), v, check_finite=False))
            lhs_l = A.__isub__(B.__imatmul__(scp.linalg.lu_solve((Dlu, Dpiv), C, check_finite=False)))
            y = scp.linalg.lu_solve(scp.linalg.lu_factor(lhs_l, check_finite=False, overwrite_a=True), rhs_l, check_finite=False, overwrite_b=True)
            solution_now = np.empty(x_shape)
            solution_now[:, 0] = y.reshape(x_shape[0], x_shape[2], x_shape[3])
            solution_now[:, 3] = scp.linalg.lu_solve((Dlu, Dpiv), v.__isub__(C @ y), check_finite=False, overwrite_b=True).reshape(x_shape[0], x_shape[2], x_shape[3])
            solution_now[:, 2] = (
                mR_d - cached_einsum('lsr,smnS,LSR,lmL->rnR', XAX_k[0, 1], block_A_k[0, 1], XAX_k1[0, 1], solution_now[:, 0]).reshape(-1, 1)
                ).__imul__(inv_I.reshape(-1, 1)).reshape(x_shape[0], x_shape[2], x_shape[3]).__isub__(solution_now[:, 3])
            solution_now[:, 1] = forward_backward_sub(
                L_L_Z, 
                mR_c - cached_einsum('lsr,smnS,LSR,rnR->lmL', XAX_k[2, 2], block_A_k[2, 2], XAX_k1[2, 2], solution_now[:, 2]).reshape(-1, 1), 
                overwrite_b=True
                ).reshape(x_shape[0], x_shape[2], x_shape[3])

        except Exception as e:
            tb = traceback.extract_tb(e.__traceback__)
            last = tb[-1]
            print(f"\t⚠️ {type(e).__name__} in {last.filename},\n\tline {last.lineno}: {last.line.strip()}")
            direct_solve_failure = True

    if not dense_solve or direct_solve_failure:

        matvec_wrapper = IneqMatVecWrapper(
            XAX_k[0, 0], XAX_k[0, 1], XAX_k[2, 1], XAX_k[2, 2], XAX_k[3, 1], XAX_k[3, 3],
            block_A_k[0, 0], block_A_k[0, 1], block_A_k[2, 1], block_A_k[2, 2], block_A_k[3, 1], block_A_k[3, 3],
            XAX_k1[0, 0], XAX_k1[0, 1], XAX_k1[2, 1], XAX_k1[2, 2], XAX_k1[3, 1], XAX_k1[3, 3],
            inv_I, x_shape[0], x_shape[2], x_shape[3]
        )
        op = scp.sparse.linalg.LinearOperator(
            shape=(3 * m, 3 * m),
            matvec=matvec_wrapper.matvec,
            dtype=np.float64
        )
        local_rhs = np.empty((3, x_shape[0], x_shape[2], x_shape[3]))
        local_rhs[0] = rhs[:, 0]
        local_rhs[1] = rhs[:, 2] - cached_einsum('lsr,smnS,LSR,rnR->lmL', XAX_k[2, 2], block_A_k[2, 2],
                                                  XAX_k1[2, 2], inv_I * rhs[:, 1])
        local_rhs[2] = rhs[:, 3]
        local_rhs_norm = np.linalg.norm(local_rhs)
        local_vec = op(np.transpose(previous_solution[:, [0, 1, 3]], (1, 0, 2, 3)).flatten()).reshape(3, x_shape[0], x_shape[2], x_shape[3])
        local_rhs_norm_prime = np.linalg.norm(local_rhs - local_vec)
        use_prev_sol = (local_rhs_norm_prime < local_rhs_norm)
        if use_prev_sol:
            local_rhs -= local_vec

        solution_now, _ = scp.sparse.linalg.gcrotmk(op, local_rhs.flatten(), rtol=rtol, k=8, maxiter=50, m=25, truncate="smallest")
        solution_now = np.transpose(solution_now.reshape(3, x_shape[0], x_shape[2], x_shape[3]),
                                    (1, 0, 2, 3)) 
        
        if use_prev_sol:
            solution_now[:, 0] += previous_solution[:, 0]
            solution_now[:, 1] += previous_solution[:, 1]
            solution_now[:, 2] += previous_solution[:, 3]

        z = inv_I * (
                    rhs[:, 1] - cached_einsum('lsr,smnS,LSR,lmL->rnR', XAX_k[0, 1], block_A_k[0, 1], XAX_k1[0, 1],
                                              solution_now[:, 0])) - solution_now[:, 2]
        solution_now = np.concatenate(
            (solution_now[:, :2], z.reshape(x_shape[0], 1, x_shape[2], x_shape[3]), solution_now[:, None, 2]), axis=1)
        
    block_res_new = np.linalg.norm(block_A_k.block_local_product(XAX_k, XAX_k1, solution_now) - rhs) / norm_rhs

    if block_res_old < block_res_new:
        solution_now = previous_solution

    return solution_now, block_res_old, min(block_res_old, block_res_new), rhs, norm_rhs, direct_solve_failure


def tt_compute_primal_feasibility(lin_op_tt, bias_tt, X_tt, status):
    primal_feas = tt_rank_reduce(tt_sub(tt_fast_matrix_vec_mul(lin_op_tt, tt_reshape(X_tt, (4,)), status.eps), bias_tt),
                   min(status.op_tol, 0.1*status.mu))  # primal feasibility
    return primal_feas


def tt_compute_dual_feasibility(obj_tt, lin_op_tt_adj, Z_tt, Y_tt, T_tt, status):
    dual_feas = tt_rank_reduce(tt_sub(tt_fast_matrix_vec_mul(lin_op_tt_adj, Y_tt, status.eps),
                                      tt_rank_reduce(tt_add(tt_reshape(Z_tt, (4,)), obj_tt), status.eps)),
                               status.eps if status.ineq_status is IneqStatus.ACTIVE else min(status.op_tol,
                                                                                              0.1*status.mu))
    if status.ineq_status is IneqStatus.ACTIVE and T_tt is not None:
        dual_feas = tt_rank_reduce(tt_sub(dual_feas, tt_reshape(T_tt, (4,))), min(status.op_tol, 0.1*status.mu))
    return dual_feas


def tt_compute_centrality(X_tt, Z_tt, status):
    if status.aho_direction:
        centrality_feas = tt_reshape(tt_scale(-1, _tt_symmetrise(tt_mat_mat_mul(X_tt, Z_tt, min(status.op_tol, 0.1*status.mu), status.eps),
                                                        min(status.op_tol, 0.1*status.mu))), (4,))
    else:
        centrality_feas = tt_reshape(tt_scale(-1, tt_mat_mat_mul(Z_tt, X_tt, min(status.op_tol, 0.1*status.mu), status.eps)), (4,))
    return centrality_feas


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
    rhs = TTBlockVector()

    # Check primal feasibility and compute residual
    primal_feas = tt_compute_primal_feasibility(lin_op_tt, bias_tt, X_tt, status)
    status.primal_error = np.divide(tt_norm(primal_feas), status.primal_error_normalisation)
    status.is_primal_feasible = np.less(status.primal_error, status.feasibility_tol)

    # Check dual feasibility and compute residual
    dual_feas = tt_compute_dual_feasibility(obj_tt, lin_op_tt_adj, Z_tt, Y_tt, T_tt, status)
    status.dual_error = np.divide(tt_norm(dual_feas), status.dual_error_normalisation)
    status.is_dual_feasible = np.less(status.dual_error, (1 + (status.ineq_status is IneqStatus.ACTIVE))*status.feasibility_tol)

    status.is_last_iter = status.is_last_iter or (status.is_primal_feasible and status.is_dual_feasible and status.is_central)

    if status.aho_direction:
        if status.is_last_iter:
            lhs[2, 1] = tt_rank_reduce(tt_scale(0.5, tt_add(tt_IkronM(Z_tt), tt_MkronI(Z_tt))), eps=status.op_tol)
            lhs[2, 2] = tt_rank_reduce(tt_scale(0.5, tt_add(tt_MkronI(X_tt), tt_IkronM(X_tt))), eps=status.op_tol)
        else:
            lhs[2, 1] = tt_psd_rank_reduce(tt_scale(0.5, tt_add(tt_IkronM(Z_tt), tt_MkronI(Z_tt))), eps=status.op_tol)
            lhs[2, 2] = tt_psd_rank_reduce(tt_scale(0.5, tt_add(tt_MkronI(X_tt), tt_IkronM(X_tt))), eps=status.op_tol)
    else:
        if status.is_last_iter:
            lhs[2, 1] = tt_rank_reduce(tt_MkronI(Z_tt), eps=status.op_tol)
            lhs[2, 2] = tt_rank_reduce(tt_IkronM(X_tt), eps=status.op_tol)
        else:
            lhs[2, 1] = tt_psd_rank_reduce(tt_MkronI(Z_tt), eps=status.op_tol)
            lhs[2, 2] = tt_psd_rank_reduce(tt_IkronM(X_tt), eps=status.op_tol)

    if not status.is_primal_feasible or status.is_last_iter:
        rhs[0] = primal_feas

    if not status.is_dual_feasible or status.is_last_iter:
        rhs[1] = dual_feas

    if not status.is_central or status.is_last_iter:
        rhs[2] = tt_compute_centrality(X_tt, Z_tt, status)

    if status.ineq_status is IneqStatus.ACTIVE:
        lhs[3, 1] =  tt_diag_op(T_tt, status.op_tol)
        masked_X_tt = tt_rank_reduce(tt_add(tt_scale(status.ineq_boundary_val, ineq_mask), tt_fast_hadamard(ineq_mask, X_tt, status.eps)), eps=status.eps)
        lhs[3, 3] = tt_rank_reduce(tt_add(status.lag_map_t, tt_diag_op(masked_X_tt, status.eps)), eps=status.op_tol)
        if not status.is_central or status.is_last_iter:
            rhs[3] = tt_rank_reduce(tt_reshape(tt_scale(-1, tt_fast_hadamard(masked_X_tt, T_tt, status.eps)), (4, )), eps=min(status.op_tol, 0.1*status.mu))

    return lhs, rhs, status

def _tt_symmetrise(matrix_tt, err_bound):
    return tt_rank_reduce(tt_scale(0.5, tt_add(matrix_tt, tt_transpose(matrix_tt))), eps=err_bound)

def _tt_psd_symmetrise(matrix_tt, err_bound):
    return tt_psd_rank_reduce(tt_scale(0.5, tt_add(matrix_tt, tt_transpose(matrix_tt))), eps=err_bound)


def _tt_mask_symmetrise(matrix_tt, mask_tt, err_bound):
    return tt_mask_rank_reduce(tt_scale(0.5, tt_add(matrix_tt, tt_transpose(matrix_tt))), mask_tt, eps=err_bound)

def _tt_get_block(i, block_matrix_tt):
    b = np.argmax([len(c.shape) for c in block_matrix_tt])
    return block_matrix_tt[:b] + [block_matrix_tt[b][:, i]] + block_matrix_tt[b+1:]

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
    try:
        # Predictor
        if status.verbose:
            print("\n--- Predictor  step ---", flush=True)
        Delta_tt, res = solver(lhs_matrix_tt, rhs_vec_tt, status.mals_delta0, status.kkt_iterations + status.is_last_iter, status.mals_rank_restriction, status.eta)
        status.mals_delta0 = Delta_tt
        Delta_X_tt = _tt_symmetrise(tt_reshape(_tt_get_block(1, Delta_tt), (2, 2)), status.eps)
        Delta_Z_tt = _tt_symmetrise(tt_reshape(_tt_get_block(2, Delta_tt), (2, 2)), status.eps)
        Delta_Y_tt = tt_rank_reduce(_tt_get_block(0, Delta_tt), eps=status.eps)
        Delta_T_tt = None
        if status.ineq_status is IneqStatus.ACTIVE:
            Delta_T_tt = tt_rank_reduce(_tt_get_block(3, Delta_tt), eps=status.eps)
            Delta_T_tt = tt_fast_hadamard(ineq_mask, tt_reshape(Delta_T_tt, (2, 2)), status.eps)

        x_step_size, z_step_size = _tt_get_step_sizes(
            X_tt,
            Z_tt,
            T_tt,
            Delta_X_tt,
            Delta_Z_tt,
            Delta_T_tt,
            ineq_mask,
            status
        )

        if not status.is_central and not status.is_last_iter:

            DXZ = tt_inner_prod(Delta_X_tt, Delta_Z_tt)
            # Corrector
            if status.verbose:
                print(f"\n--- Centering-Corrector  step ---", flush=True)

            if status.ineq_status is IneqStatus.ACTIVE:
                mu_aff = (
                    ZX + x_step_size * z_step_size * DXZ
                    + z_step_size * tt_inner_prod(X_tt, Delta_Z_tt)
                    + x_step_size * tt_inner_prod(Delta_X_tt, Z_tt)
                    + TX + x_step_size * z_step_size * tt_inner_prod(Delta_T_tt, Delta_X_tt)
                    + z_step_size * (tt_inner_prod(X_tt, Delta_T_tt) + status.ineq_boundary_val*tt_entrywise_sum(Delta_T_tt))
                    + x_step_size * tt_inner_prod(Delta_X_tt, T_tt)
                )
                e = max(1, 3 * min(x_step_size, z_step_size) ** 2)
                status.sigma = min(0.99, (mu_aff/(ZX + TX))**e)
                rhs_3 = tt_add(
                        tt_scale(status.sigma * status.mu, tt_reshape(ineq_mask, (4,))),
                        rhs_vec_tt.get_row(3)
                        ) if status.sigma > 0 else rhs_vec_tt.get_row(3)
                rhs_vec_tt[3] = tt_rank_reduce(
                    rhs_3,
                    min(status.op_tol, 0.1*status.mu)
            )
            else:
                mu_aff = (
                    ZX + x_step_size * z_step_size * DXZ
                    + z_step_size * tt_inner_prod(X_tt,Delta_Z_tt)
                    + x_step_size * tt_inner_prod(Delta_X_tt, Z_tt)
                )
                e = max(1, 3*min(x_step_size, z_step_size)**2)
                status.sigma = min(0.99, (mu_aff/ZX) ** e)


            if DXZ > 0.1*status.centrality_tol:
                Delta_XZ_term = tt_compute_centrality(Delta_X_tt, Delta_Z_tt, status)
                rhs_vec_tt[2] = tt_rank_reduce(
                    tt_add(
                        tt_scale(status.sigma * status.mu, tt_reshape(tt_identity(len(X_tt)), (4,))),
                        tt_add(
                            rhs_vec_tt.get_row(2),
                            Delta_XZ_term
                        )
                    ),
                    min(status.op_tol, 0.1*status.mu)
                ) if status.sigma > 1e-4 else tt_rank_reduce(tt_add(rhs_vec_tt.get_row(2), Delta_XZ_term), min(status.op_tol, 0.1*status.mu))
            else:
                rhs_vec_tt[2] = tt_rank_reduce(
                    tt_add(
                        tt_scale(status.sigma * status.mu, tt_reshape(tt_identity(len(X_tt)), (4,))),
                        rhs_vec_tt.get_row(2)
                    ),
                    min(status.op_tol, 0.1*status.mu)
                ) if status.sigma > 1e-4 else rhs_vec_tt.get_row(2)

            Delta_tt_cc, res = solver(lhs_matrix_tt, rhs_vec_tt, status.mals_delta0, status.kkt_iterations + status.is_last_iter, status.mals_rank_restriction, status.eta)
            status.mals_delta0 = Delta_tt_cc
            Delta_X_tt_cc = _tt_symmetrise(tt_reshape(_tt_get_block(1, Delta_tt_cc), (2, 2)), status.eps)
            Delta_Z_tt_cc = _tt_symmetrise(tt_reshape(_tt_get_block(2, Delta_tt_cc), (2, 2)), status.eps)
            Delta_Y_tt_cc = tt_rank_reduce(_tt_get_block(0, Delta_tt_cc), eps=status.eps)
            Delta_X_tt = tt_rank_reduce(tt_add(Delta_X_tt_cc, Delta_X_tt), eps=status.eps)
            Delta_Y_tt = tt_rank_reduce(tt_add(Delta_Y_tt_cc, Delta_Y_tt), eps=status.eps)
            Delta_Z_tt = tt_rank_reduce(tt_add(Delta_Z_tt_cc, Delta_Z_tt), eps=status.eps)
            if status.ineq_status is IneqStatus.ACTIVE:
                Delta_T_tt_cc = tt_rank_reduce(_tt_get_block(3, Delta_tt_cc), eps=status.eps)
                Delta_T_tt_cc = tt_fast_hadamard(ineq_mask, tt_reshape(Delta_T_tt_cc, (2, 2)), status.eps)
                Delta_T_tt = tt_rank_reduce(tt_add(Delta_T_tt_cc, Delta_T_tt), eps=status.eps)

            x_step_size, z_step_size = _tt_get_step_sizes(
                X_tt,
                Z_tt,
                T_tt,
                Delta_X_tt,
                Delta_Z_tt,
                Delta_T_tt,
                ineq_mask,
                status
            )
        else:
            status.sigma = 0
    except Exception as e:
        print(f"\n\t⚠️ Attention: {e}")
        print("\n\t==> Full traceback (most recent call last):")
        traceback.print_exc(file=sys.stdout)
        return 0, 0, None, None, None, None, status

    return x_step_size, z_step_size, Delta_X_tt, Delta_Y_tt, Delta_Z_tt, Delta_T_tt, status


def _tt_get_step_sizes(
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
        X_tt = tt_add(X_tt, tt_scale(status.boundary_val, tt_identity(len(X_tt))))
        Z_tt = tt_add(Z_tt, tt_scale(status.boundary_val, tt_identity(len(Z_tt))))

    x_step_size, status.eigen_x0 = tt_max_generalised_eigen(X_tt, Delta_X_tt, x0=status.eigen_x0, tol=1e-7, verbose=status.verbose)
    z_step_size, status.eigen_z0 = tt_max_generalised_eigen(Z_tt, Delta_Z_tt, x0=status.eigen_z0, tol=1e-7, verbose=status.verbose)
    if status.ineq_status is not IneqStatus.NOT_IN_USE:
        if status.is_last_iter:
            X_tt = tt_add(X_tt, tt_scale(status.ineq_boundary_val + status.boundary_val, ineq_mask))
            T_tt = tt_add(T_tt, tt_scale(status.ineq_boundary_val + status.boundary_val, ineq_mask))
        x_step_size, z_step_size = _tt_get_ineq_step_sizes(x_step_size, z_step_size, X_tt, T_tt, Delta_X_tt, Delta_T_tt, ineq_mask, status)
    tau_x = 0.9 + 0.05*min(x_step_size,  z_step_size) if x_step_size < 1 else 1.0
    tau_z = 0.9 + 0.05*min(x_step_size,  z_step_size) if z_step_size < 1 else 1.0

    if status.verbose:
        print(f"Step search concluded.")
        print(f"Step sizes: a_p:{x_step_size:.2e}, a_d:{z_step_size:.2e}")
    return tau_x*x_step_size, tau_z*z_step_size


def _ineq_step_size(A_tt, Delta_tt, e_tt, status):
    sum_tt = tt_add(A_tt, Delta_tt)
    if status.compl_ineq_mask:
        sum_tt = tt_add(sum_tt, tt_scale(tt_entrywise_sum(sum_tt)/status.num_ineq_constraints, status.compl_ineq_mask))
    sum_tt = tt_rank_reduce(sum_tt, status.eps)
    e_tt, _ = tt_min_eig(tt_diag_op(sum_tt, status.eps), x0=e_tt, tol=1e-7, verbose=status.verbose)
    e_tt_sq = tt_reshape(tt_normalise(tt_fast_hadamard(e_tt, e_tt, status.eps)), (2, 2))
    min_A_val = tt_inner_prod(A_tt, e_tt_sq)
    min_Delta_val = tt_inner_prod(Delta_tt, e_tt_sq)
    if min_Delta_val >= 0:
        step_size = 1
    else:
        step_size = np.clip(-(1-status.op_tol)*min_A_val/min_Delta_val, a_min=0, a_max=1)
    return step_size, e_tt


def _tt_get_ineq_step_sizes(x_step_size, z_step_size, X_tt, T_tt, Delta_X_tt, Delta_T_tt, ineq_mask, status):

    if x_step_size > 0:
        masked_X_tt = tt_fast_hadamard(ineq_mask, X_tt, status.eps)
        masked_Delta_X_tt = tt_fast_hadamard(ineq_mask, Delta_X_tt, status.eps)
        x_ineq_step_size, status.eigen_xt0 = _ineq_step_size(
            tt_add(masked_X_tt, tt_scale(status.ineq_boundary_val, ineq_mask)),
            tt_scale(x_step_size, masked_Delta_X_tt),
            status.eigen_xt0,
            status
        )
        if not status.is_last_iter:
            if 1 - x_ineq_step_size < status.eps and tt_norm(T_tt) < 0.5*status.op_tol:
                if status.ineq_status is IneqStatus.ACTIVE:
                    status.ineq_status = IneqStatus.SETTING_INACTIVE
            else:
                if status.ineq_status is IneqStatus.INACTIVE:
                    status.ineq_status = IneqStatus.SETTING_ACTIVE
        x_step_size *= x_ineq_step_size

    if z_step_size > 0 and status.ineq_status is not IneqStatus.INACTIVE:
        t_step_size, status.eigen_zt0 = _ineq_step_size(
            T_tt,
            tt_scale(z_step_size, Delta_T_tt),
            status.eigen_zt0,
        status
        )
        z_step_size *= t_step_size

    return x_step_size, z_step_size


def _initialise(ineq_mask, status, dim, epsilonDash):
    X_tt = tt_scale(epsilonDash, tt_identity(dim))
    Z_tt = tt_scale(epsilonDash, tt_identity(dim))
    Y_tt = tt_reshape(tt_zero_matrix(dim), (4, ))
    T_tt = None

    if status.ineq_status is IneqStatus.ACTIVE:
        T_tt = tt_scale(status.eps, ineq_mask)
        # Need to initialise so it stays psd
        X_tt = tt_rank_reduce(tt_add(X_tt, tt_scale(1/2**(dim/2), ineq_mask)), status.op_tol)

    return X_tt, Y_tt, Z_tt, T_tt

@dataclass
class IPMStatus:
    dim: int
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
    centrality_error: float
    mu: float

    is_last_iter: bool
    ineq_status: IneqStatus
    verbose: bool

    primal_error_normalisation: float
    dual_error_normalisation: float
    mals_rank_restriction: int

    boundary_val: float = 1e-10
    ineq_boundary_val: float = 0.01
    sigma: float = 0.5
    num_ineq_constraints: float = 0
    lag_map_t = None
    lag_map_y = None
    compl_ineq_mask = None
    mals_delta0 = None
    eigen_x0 = None
    eigen_z0 = None
    eigen_xt0 = None
    eigen_zt0 = None
    kkt_iterations = 8
    centrl_error_normalisation: float = 1.0
    eta = 1e-3


def _ipm_format_output(X_tt, Y_tt, T_tt, Z_tt, iteration, dim):
    """Formats the final results into the desired output structure."""
    ranksX = tt_ranks(X_tt)
    ranksZ = tt_ranks(Z_tt)
    ranksY = tt_ranks(Y_tt)
    ranksT = tt_ranks(T_tt) if T_tt else [0] * (dim - 1)
    
    print("---Terminated---")
    print(f"Converged in {iteration} iterations.")
    print(f"Ranks: X={ranksX}, Z={ranksZ}, Y={ranksY}, T={ranksT}")
    
    results = {"num_iters": iteration, "ranksX": ranksX, "ranksY": ranksY, "ranksZ": ranksZ, "ranksT": ranksT}
    return X_tt, Y_tt, T_tt, Z_tt, results


def _ipm_check_for_stalled_progress(prev_errors, status, gap_tol):
    """Checks if the optimization has stalled."""
    if status.is_last_iter:
        return False
        
    primal_stalled = abs(prev_errors['primal'] - status.primal_error) < 0.04 * gap_tol
    dual_stalled = abs(prev_errors['dual'] - status.dual_error) < 0.04 * gap_tol
    centrality_stalled = abs(prev_errors['centrality'] - status.centrality_error) < 0.02 * gap_tol
    
    if primal_stalled and dual_stalled and centrality_stalled:
        if status.verbose:
            print("============================================\n Progress stalled! Entering finishing phase.\n============================================")
        return True
    return False


def _ipm_check_convergence(status, finishing_steps, ZX, TX, abs_tol, max_refinement):
    """Checks for final convergence and updates the finishing step counter."""
    if not status.is_last_iter:
        return status, finishing_steps
        
    converged = (abs(ZX) + abs(TX) <= abs_tol and 
                 status.primal_error < abs_tol and 
                 status.dual_error < abs_tol)
    if converged:
        finishing_steps = 0
    else:
        finishing_steps -= 1
        status.boundary_val = 0.01 * (1 - (finishing_steps / max_refinement))
        if finishing_steps == 1:
            status.kkt_iterations += 1
            
    return status, finishing_steps


def _ipm_log_iteration(iteration, status, X_tt, Y_tt, Z_tt, T_tt):
    """Prints verbose output for the current iteration."""
    print(f"\n--- Iteration {iteration - 1} ---")
    print(f"Status: Finishing up={status.is_last_iter}, Ineq={str(status.ineq_status)}")
    print(f"Feasibility: Central={status.is_central}, Primal={status.is_primal_feasible}, Dual={status.is_dual_feasible}")
    print(f"Direction: {'AHO' if status.aho_direction else 'XZ'}, Sigma: {status.sigma:.2e}")
    print(f"Errors: Centrality={status.centrality_error:.4e}, Primal={status.primal_error:.4e}, Dual={status.dual_error:.4e}")
    print(f"Ranks: X={tt_ranks(X_tt)}, Z={tt_ranks(Z_tt)}, Y={tt_ranks(Y_tt)}, T={tt_ranks(T_tt) if T_tt else 'N/A'}")


def tt_ipm(
    lag_maps,
    obj_tt,
    lin_op_tt,
    bias_tt,
    ineq_mask=None,
    max_iter=100,
    max_refinement=5,
    warm_up=3, #8
    gap_tol=1e-4,
    aho_direction=True,
    op_tol=1e-5,
    abs_tol=5e-4,
    eps=1e-12,
    mals_restarts=3,
    r_max=700,
    epsilonDash=1,
    verbose=False
):
    centrality_tol = gap_tol
    feasibility_tol = 2*gap_tol
    dim = len(obj_tt)
    status = IPMStatus(
        len(obj_tt),
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
        np.inf,
        False,
        IneqStatus.NOT_IN_USE if ineq_mask is None else IneqStatus.ACTIVE,
        verbose,
        1,
        1,
        r_max
    )
    lag_maps = {key: tt_rank_reduce(value, eps=eps) for key, value in lag_maps.items()}
    obj_tt = tt_rank_reduce(obj_tt, eps=eps)
    lin_op_tt = tt_rank_reduce(lin_op_tt, eps=eps)
    bias_tt = tt_rank_reduce(bias_tt, eps=eps)

    status.primal_error_normalisation = 1 + tt_norm(bias_tt)
    status.dual_error_normalisation = 1 + tt_norm(obj_tt)

    lhs_skeleton = TTBlockMatrix()
    lhs_skeleton[1, 2] = tt_reshape(tt_identity(2 * dim), (4, 4))
    solver_ineq = lambda lhs, rhs, x0, nwsp, restriction, termination_tol: tt_restarted_block_amen(
        lhs,
        rhs,
        rank_restriction=restriction, # max(4*dim + dim + 4, 25)
        x0=x0,
        local_solver=_ipm_local_solver_ineq,
        op_tol=op_tol,
        termination_tol=termination_tol,
        num_restarts=mals_restarts, # 3
        inner_m=nwsp,
        verbose=verbose
    )
    solver_eq = lambda lhs, rhs, x0, nwsp, restriction, termination_tol: tt_restarted_block_amen(
        lhs,
        rhs,
        rank_restriction=restriction, # max(3*dim + dim + 3, 25)
        x0=x0,
        local_solver=_ipm_local_solver,
        op_tol=op_tol,
        termination_tol=termination_tol,
        num_restarts=mals_restarts, # 3
        inner_m=nwsp,
        verbose=verbose
    )
    if status.ineq_status is IneqStatus.ACTIVE:
        solver = solver_ineq
        status.num_ineq_constraints = tt_inner_prod(ineq_mask, ineq_mask)
        status.compl_ineq_mask = tt_rank_reduce(tt_sub(tt_one_matrix(dim), ineq_mask), eps=eps)
        status.lag_map_t = lag_maps["t"]
        lhs_skeleton.add_alias((1, 2), (1, 3))
    else:
        solver = solver_eq
        status.num_ineq_constraints = 0

    # KKT-system prep
    lin_op_tt_adj = tt_transpose(lin_op_tt)
    lhs_skeleton[0, 1] = tt_scale(-1, lin_op_tt)
    lhs_skeleton.add_alias((0, 1), (1, 0), is_transpose=True) #lhs_skeleton[1, 0] = tt_scale(-1, lin_op_tt_adj)
    lhs_skeleton[0, 0] = lag_maps["y"]
    status.lag_map_y = lag_maps["y"]

    X_tt, Y_tt, Z_tt, T_tt = _initialise(ineq_mask, status, dim, epsilonDash)

    iteration = 0
    finishing_steps = max_refinement
    prev_errors = {'primal': np.inf, 'dual': np.inf, 'centrality': np.inf}
    lhs = lhs_skeleton

    while finishing_steps > 0:
        iteration += 1
        status.aho_direction = (iteration > warm_up)
        status.is_last_iter = status.is_last_iter or (max_iter - max_refinement < iteration)
        ZX = tt_inner_prod(Z_tt, X_tt)
        TX = tt_inner_prod(X_tt, T_tt) + status.ineq_boundary_val*tt_entrywise_sum(T_tt) if status.ineq_status is IneqStatus.ACTIVE else 0
        status.mu = np.divide(abs(ZX) + abs(TX), (2 ** dim + (status.ineq_status is IneqStatus.ACTIVE)*status.num_ineq_constraints))
        status.centrl_error_normalisation = 1 + abs(tt_inner_prod(obj_tt, tt_reshape(X_tt, (4, ))))
        status.centrality_error = status.mu / status.centrl_error_normalisation
        status.is_central = np.less(status.centrality_error, (1 + (status.ineq_status is IneqStatus.ACTIVE))*centrality_tol)
        status.eta = max(min(status.eta, 2*status.mu), status.op_tol)

        lhs_matrix_tt, rhs_vec_tt, status = tt_infeasible_newton_system(
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
        )

        if verbose:
            _ipm_log_iteration(iteration, status, X_tt, Y_tt, Z_tt, T_tt)

        status, finishing_steps = _ipm_check_convergence(
            status, finishing_steps, ZX, TX, abs_tol, max_refinement
        )
        if finishing_steps == 0:
            iteration -= 1
            break

        x_step_size, z_step_size, Delta_X_tt, Delta_Y_tt, Delta_Z_tt, Delta_T_tt, status = _tt_ipm_newton_step(
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


        if (Delta_X_tt is None and Delta_Z_tt is None) or (x_step_size < 1e-5 and z_step_size < 1e-5):
            if status.is_last_iter:
                break
            else:
                status.is_last_iter = True
        else:
            if status.is_last_iter:
                X_tt = _tt_symmetrise(tt_add(X_tt, tt_scale(x_step_size, Delta_X_tt)), status.op_tol)
            else:
                X_tt = _tt_psd_symmetrise(tt_add(X_tt, tt_scale(x_step_size, Delta_X_tt)), status.op_tol)
            if status.is_last_iter:
                Z_tt = _tt_symmetrise(tt_add(Z_tt, tt_scale(z_step_size, Delta_Z_tt)), status.op_tol)
            else:
                Z_tt = _tt_psd_symmetrise(tt_add(Z_tt, tt_scale(z_step_size, Delta_Z_tt)), status.op_tol)

            Y_tt = tt_rank_reduce(tt_add(Y_tt, tt_scale(z_step_size, Delta_Y_tt)), status.eps)
            Y_tt = tt_reshape(_tt_symmetrise(tt_reshape(tt_sub(Y_tt, tt_fast_matrix_vec_mul(status.lag_map_y, Y_tt, status.eps)), (2, 2)), status.op_tol), (4, ))

            if status.ineq_status is IneqStatus.ACTIVE:
                if status.is_last_iter:
                    T_tt = _tt_symmetrise(tt_add(T_tt, tt_scale(z_step_size, Delta_T_tt)), op_tol)
                else:
                    T_tt = _tt_mask_symmetrise(tt_add(T_tt, tt_scale(z_step_size, Delta_T_tt)), ineq_mask, op_tol)
            elif status.ineq_status is IneqStatus.SETTING_INACTIVE:
                solver = solver_eq
                lhs = lhs_skeleton.get_submatrix(2, 2)
                status.mals_delta0 = None
                status.ineq_status = IneqStatus.INACTIVE
                T_tt = tt_scale(status.eps, ineq_mask)
            elif status.ineq_status is IneqStatus.SETTING_ACTIVE:
                solver = solver_ineq
                lhs = lhs_skeleton
                status.mals_delta0 = None
                status.ineq_status = IneqStatus.ACTIVE

        if _ipm_check_for_stalled_progress(prev_errors, status, gap_tol):
            status.is_last_iter = True

        prev_errors['primal'] = status.primal_error
        prev_errors['dual'] = status.dual_error
        prev_errors['centrality'] = status.centrality_error

    return _ipm_format_output(X_tt, Y_tt, T_tt, Z_tt, iteration, status.dim)