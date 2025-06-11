import sys
import os

sys.path.append(os.getcwd() + '/../')

from src.tt_ops import *
from src.tt_als import cached_einsum, TTBlockMatrix, TTBlockVector, tt_restarted_block_als, tt_max_generalised_eigen, tt_min_eig, tt_mat_mat_mul
from dataclasses import dataclass
from enum import Enum
from src.tt_ops import lgmres

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

def _ipm_local_solver(XAX_k, block_A_k, XAX_k1, Xb_k, block_b_k, Xb_k1, previous_solution, size_limit, termination_tol):
    x_shape = previous_solution.shape
    m = x_shape[0] * x_shape[2] * x_shape[3]
    rhs = np.zeros_like(previous_solution)
    rhs[:, 0] = cached_einsum('br,bmB,BR->rmR', Xb_k[0], block_b_k[0], Xb_k1[0]) if 0 in block_b_k else 0
    rhs[:, 1] = cached_einsum('br,bmB,BR->rmR', Xb_k[1], block_b_k[1], Xb_k1[1]) if 1 in block_b_k else 0
    rhs[:, 2] = cached_einsum('br,bmB,BR->rmR', Xb_k[2], block_b_k[2], Xb_k1[2]) if 2 in block_b_k else 0
    inv_I = np.divide(1, cached_einsum('lsr,smnS,LSR->lmL', XAX_k[1, 2], block_A_k[1, 2], XAX_k1[1, 2]))
    block_res_old = np.linalg.norm(block_A_k.block_local_product(XAX_k, XAX_k1, previous_solution).__isub__(rhs))
    size_limit = 0
    if m <= size_limit:
        try:
            L_L_Z = scp.linalg.cholesky(
                cached_einsum('lsr,smnS,LSR->lmLrnR', XAX_k[2, 1], block_A_k[2, 1], XAX_k1[2, 1]).reshape(m, m),
                check_finite=False, lower=True, overwrite_a=True
            )
            mR_p = rhs[:, 0].reshape(m, 1)
            mR_d = rhs[:, 1].reshape(m, 1)
            mR_c = rhs[:, 2].reshape(m, 1)
            L_X = cached_einsum('lsr,smnS,LSR->lmLrnR', XAX_k[2, 2], block_A_k[2, 2], XAX_k1[2, 2]).reshape(m, m)
            mL_eq = cached_einsum('lsr,smnS,LSR->lmLrnR', XAX_k[0, 1], block_A_k[0, 1], XAX_k1[0, 1]).reshape(m, m)
            A = (mL_eq @ forward_backward_sub(L_L_Z, L_X * inv_I.reshape(1, -1), overwrite_b=True) @ mL_eq.T).__iadd__(cached_einsum('lsr,smnS,LSR->lmLrnR',XAX_k[0, 0], block_A_k[0, 0], XAX_k1[0, 0]).reshape(m, m))
            b = mR_p - mL_eq @ forward_backward_sub(L_L_Z, mR_c - (L_X * inv_I.reshape(1, -1)) @ mR_d, overwrite_b=True) - A @ previous_solution[:, 0].reshape(-1, 1)
            solution_now = np.empty(x_shape)
            solution_now[:, 0] = scp.linalg.solve(A, b, check_finite=False, overwrite_a=True, overwrite_b=True).reshape(x_shape[0], x_shape[2], x_shape[3]).__iadd__(previous_solution[:, 0])
            solution_now[:, 2] = (mR_d - mL_eq.T @ solution_now[:, 0].reshape(-1, 1)).__imul__(inv_I.reshape(-1, 1)).reshape(x_shape[0], x_shape[2], x_shape[3])
            solution_now[:, 1] = forward_backward_sub(L_L_Z, mR_c - L_X @ solution_now[:, 2].reshape(-1, 1), overwrite_b=True).reshape(x_shape[0], x_shape[2], x_shape[3])
        except Exception as e:
            print(f"\tAttention: {e}")
            size_limit = 0

    if m > size_limit:
        Op = MatVecWrapper(
            XAX_k[0, 0], XAX_k[0, 1], XAX_k[2, 1], XAX_k[2, 2],
            block_A_k[0, 0], block_A_k[0, 1], block_A_k[2, 1], block_A_k[2, 2],
            XAX_k1[0, 0], XAX_k1[0, 1], XAX_k1[2, 1], XAX_k1[2, 2],
            inv_I, x_shape[0], x_shape[2], x_shape[3]
        )

        local_rhs = -Op.matvec(np.transpose(previous_solution[:, :2], (1, 0, 2, 3)).ravel()).reshape(2, x_shape[0], x_shape[2], x_shape[3])
        local_rhs[0] += rhs[:, 0]
        local_rhs[1] += rhs[:, 2]
        local_rhs[1] -= cached_einsum('lsr,smnS,LSR,rnR->lmL', XAX_k[2, 2], block_A_k[2, 2], XAX_k1[2, 2], inv_I*rhs[:, 1])

        max_iter = min(max(2 * int(np.ceil(block_res_old / termination_tol)), 2), 50)
        solution_now, info = lgmres(
            Op,
            local_rhs.ravel(),
            rtol=1e-10,
            outer_k=5,
            inner_m=20,
            maxiter=max_iter
        )
        solution_now = np.transpose(solution_now.reshape(2, x_shape[0], x_shape[2], x_shape[3]), (1, 0, 2, 3)).__iadd__(previous_solution[:, :2])

        z = inv_I * (rhs[:, 1] - cached_einsum('lsr,smnS,LSR,lmL->rnR', XAX_k[0, 1], block_A_k[0, 1], XAX_k1[0, 1], solution_now[:, 0]))
        solution_now = np.concatenate((solution_now, z.reshape(x_shape[0], 1, x_shape[2], x_shape[3])), axis=1)

    block_res_new = np.linalg.norm(block_A_k.block_local_product(XAX_k, XAX_k1, solution_now).__isub__(rhs))

    if block_res_old < block_res_new:
        solution_now = previous_solution

    return solution_now, block_res_old, min(block_res_new, block_res_old), rhs

def _ipm_local_solver_ineq(XAX_k, block_A_k, XAX_k1, Xb_k, block_b_k, Xb_k1, previous_solution, size_limit, termination_tol):
    x_shape = previous_solution.shape
    m = x_shape[0] * x_shape[2] * x_shape[3]
    rhs = np.zeros_like(previous_solution)
    rhs[:, 0] = cached_einsum('br,bmB,BR->rmR', Xb_k[0], block_b_k[0], Xb_k1[0]) if 0 in block_b_k else 0
    rhs[:, 1] = cached_einsum('br,bmB,BR->rmR', Xb_k[1], block_b_k[1], Xb_k1[1]) if 1 in block_b_k else 0
    rhs[:, 2] = cached_einsum('br,bmB,BR->rmR', Xb_k[2], block_b_k[2], Xb_k1[2]) if 2 in block_b_k else 0
    rhs[:, 3] = cached_einsum('br,bmB,BR->rmR', Xb_k[3], block_b_k[3], Xb_k1[3]) if 3 in block_b_k else 0
    inv_I = np.divide(1, cached_einsum('lsr,smnS,LSR->lmL', XAX_k[1, 2], block_A_k[1, 2], XAX_k1[1, 2]))
    block_res_old_scalar = np.linalg.norm(block_A_k.block_local_product(XAX_k, XAX_k1, previous_solution).__isub__(rhs))
    if m <= size_limit:
        try:
            L_L_Z = scp.linalg.cholesky(
                cached_einsum('lsr,smnS,LSR->lmLrnR', XAX_k[2, 1], block_A_k[2, 1], XAX_k1[2, 1]).reshape(m, m),
                check_finite=False, lower=True,  overwrite_a=True
            )
            mR_p = rhs[:, 0].reshape(m, 1)
            mR_d = rhs[:, 1].reshape(m, 1)
            mR_t = rhs[:, 3].reshape(m, 1)
            L_L_Z_inv_mR_c = forward_backward_sub(L_L_Z, rhs[:, 2].reshape(m, 1))
            L_X = cached_einsum('lsr,smnS,LSR->lmLrnR', XAX_k[2, 2], block_A_k[2, 2], XAX_k1[2, 2]).reshape(m, m)
            mL_eq = cached_einsum('lsr,smnS,LSR->lmLrnR', XAX_k[0, 1], block_A_k[0, 1], XAX_k1[0, 1]).reshape(m, m)
            T_op = cached_einsum('lsr,smnS,LSR->lmLrnR', XAX_k[3, 1], block_A_k[3, 1], XAX_k1[3, 1]).reshape(m, m)
            A = cached_einsum('lsr,smnS,LSR->lmLrnR', XAX_k[0, 0], block_A_k[0, 0],XAX_k1[0, 0]).reshape(m, m).__iadd__(mL_eq @ forward_backward_sub(L_L_Z, L_X * inv_I.reshape(1, -1) @ mL_eq.T, overwrite_b=True))
            B = mL_eq @ forward_backward_sub(L_L_Z, L_X)
            C = T_op @ forward_backward_sub(L_L_Z, (L_X * inv_I.reshape(1, -1)) @ mL_eq.T, overwrite_b=True)
            D = cached_einsum('lsr,smnS,LSR->lmLrnR', XAX_k[3, 3], block_A_k[3, 3], XAX_k1[3, 3]).reshape(m, m).__iadd__(T_op @ forward_backward_sub(L_L_Z, L_X))

            u = (
                    mR_p - mL_eq @ (L_L_Z_inv_mR_c - forward_backward_sub(L_L_Z, (L_X * inv_I.reshape(1, -1)) @ mR_d, overwrite_b=True))
                    - A @ previous_solution[:, 0].reshape(-1, 1)
                    - B @ previous_solution[:, 3].reshape(-1, 1)
            )
            v = (
                    mR_t - T_op @ (L_L_Z_inv_mR_c - forward_backward_sub(L_L_Z, (L_X * inv_I.reshape(1, -1)) @ mR_d, overwrite_b=True))
                    - C @ previous_solution[:, 0].reshape(-1, 1)
                    - D @ previous_solution[:, 3].reshape(-1, 1)
            )
            Dlu, Dpiv = scp.linalg.lu_factor(D, check_finite=False, overwrite_a=True)
            rhs_l = u.__isub__(B @ scp.linalg.lu_solve((Dlu, Dpiv), v, check_finite=False))
            lhs_l = A.__isub__(B.__imatmul__(scp.linalg.lu_solve((Dlu, Dpiv), C, check_finite=False)))
            y = scp.linalg.lu_solve(scp.linalg.lu_factor(lhs_l, check_finite=False, overwrite_a=True), rhs_l, check_finite=False, overwrite_b=True)
            t = scp.linalg.lu_solve((Dlu, Dpiv), v.__isub__(C @ y), check_finite=False, overwrite_b=True)
            y += previous_solution[:, 0].reshape(-1, 1)
            t += previous_solution[:, 3].reshape(-1, 1)
            z = (inv_I.reshape(-1, 1) * (mR_d - mL_eq.T @ y)).__isub__(t)
            x = L_L_Z_inv_mR_c.__isub__(forward_backward_sub(L_L_Z, L_X @ z, overwrite_b=True))

            solution_now = np.transpose(
                np.vstack((y, x, z, t)).reshape(x_shape[1], x_shape[0], x_shape[2], x_shape[3]),
                (1, 0, 2, 3)
            )
        except:
            size_limit = 0

    if m > size_limit:
        linear_op = IneqMatVecWrapper(
            XAX_k[0, 0], XAX_k[0, 1], XAX_k[2, 1], XAX_k[2, 2], XAX_k[3, 1], XAX_k[3, 3],
            block_A_k[0, 0], block_A_k[0, 1], block_A_k[2, 1], block_A_k[2, 2], block_A_k[3, 1], block_A_k[3, 3],
            XAX_k1[0, 0], XAX_k1[0, 1], XAX_k1[2, 1], XAX_k1[2, 2], XAX_k1[3, 1], XAX_k1[3, 3],
            inv_I, x_shape[0], x_shape[2], x_shape[3]
        )
        local_rhs = -linear_op.matvec(np.transpose(previous_solution[:, [0, 1, 3]], (1, 0, 2, 3)).ravel()).reshape(3, x_shape[0], x_shape[2], x_shape[3])
        local_rhs[0] += rhs[:, 0]
        local_rhs[1] += rhs[:, 2] - cached_einsum('lsr,smnS,LSR,rnR->lmL', XAX_k[2, 2], block_A_k[2, 2],
                                                  XAX_k1[2, 2], inv_I * rhs[:, 1])
        local_rhs[2] += rhs[:, 3]
        max_iter = min(max(2 * int(np.ceil(block_res_old_scalar / termination_tol)), 2), 50)
        solution_now, _ = lgmres(
            linear_op,
            local_rhs.ravel(),
            rtol=1e-10,
            outer_k=5,
            inner_m=20,
            maxiter=max_iter
        )
        solution_now = np.transpose(solution_now.reshape(3, x_shape[0], x_shape[2], x_shape[3]),
                                    (1, 0, 2, 3)) + previous_solution[:, [0, 1, 3]]
        z = inv_I * (
                    rhs[:, 1] - cached_einsum('lsr,smnS,LSR,lmL->rnR', XAX_k[0, 1], block_A_k[0, 1], XAX_k1[0, 1],
                                              solution_now[:, 0])) - solution_now[:, 2]
        solution_now = np.concatenate(
            (solution_now[:, :2], z.reshape(x_shape[0], 1, x_shape[2], x_shape[3]), solution_now[:, None, -1]), axis=1)

    block_res_new_scalar = np.linalg.norm(block_A_k.block_local_product(XAX_k, XAX_k1, solution_now) - rhs)

    if block_res_old_scalar < block_res_new_scalar:
        solution_now = previous_solution

    return solution_now, block_res_old_scalar, min(block_res_new_scalar, block_res_old_scalar), rhs


def tt_compute_primal_feasibility(lin_op_tt, bias_tt, X_tt, status):
    primal_feas = tt_rank_reduce(tt_sub(tt_fast_matrix_vec_mul(lin_op_tt, tt_reshape(X_tt, (4,)), status.eps), bias_tt),
                   0.1*min(status.op_tol, status.feasibility_tol))  # primal feasibility
    return primal_feas


def tt_compute_dual_feasibility(obj_tt, lin_op_tt_adj, Z_tt, Y_tt, T_tt, status):
    dual_feas = tt_rank_reduce(tt_sub(tt_fast_matrix_vec_mul(lin_op_tt_adj, Y_tt, status.eps),
                                      tt_rank_reduce(tt_add(tt_reshape(Z_tt, (4,)), obj_tt), status.eps)),
                               status.eps if status.ineq_status is IneqStatus.ACTIVE else 0.1*min(status.op_tol,
                                                                                              status.feasibility_tol))
    if status.ineq_status is IneqStatus.ACTIVE and T_tt is not None:
        dual_feas = tt_rank_reduce(tt_sub(dual_feas, tt_reshape(T_tt, (4,))), 0.1*min(status.op_tol, status.feasibility_tol))
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
        masked_X_tt = tt_rank_reduce(tt_add(tt_scale(status.boundary_val, ineq_mask), tt_fast_hadammard(ineq_mask, X_tt, status.eps)), eps=status.eps)
        lhs[3, 3] = tt_rank_reduce(tt_add(status.lag_map_t, tt_diag_op(masked_X_tt, status.eps)), eps=status.op_tol)
        if not status.is_central or status.is_last_iter:
            rhs[3] = tt_rank_reduce(tt_reshape(tt_scale(-1, tt_fast_hadammard(masked_X_tt, T_tt, status.eps)), (4, )), eps=min(status.op_tol, 0.1*status.mu))

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

    # Predictor
    if status.verbose:
        print("\n--- Predictor  step ---", flush=True)
    Delta_tt, res = solver(lhs_matrix_tt, rhs_vec_tt, status.mals_delta0, status.kkt_iterations, status.is_last_iter, status.eta)
    status.mals_delta0 = Delta_tt
    Delta_X_tt = _tt_symmetrise(tt_reshape(_tt_get_block(1, Delta_tt), (2, 2)), status.eps)
    Delta_Z_tt = _tt_symmetrise(tt_reshape(_tt_get_block(2, Delta_tt), (2, 2)), status.eps)
    Delta_Y_tt = tt_rank_reduce(_tt_get_block(0, Delta_tt), eps=status.eps)
    Delta_Y_tt = tt_rank_reduce(tt_sub(Delta_Y_tt, tt_fast_matrix_vec_mul(status.lag_map_y, Delta_Y_tt, status.eps)),
                                eps=status.eps)
    Delta_T_tt = None
    if status.ineq_status is IneqStatus.ACTIVE:
        Delta_T_tt = tt_rank_reduce(_tt_get_block(3, Delta_tt), eps=status.eps)
        Delta_T_tt = tt_fast_hadammard(ineq_mask, tt_reshape(Delta_T_tt, (2, 2)), status.eps)
    x_step_size, z_step_size = _tt_line_search(
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
                + z_step_size * (tt_inner_prod(X_tt, Delta_T_tt) + status.boundary_val*tt_entrywise_sum(Delta_T_tt))
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
                status.op_tol
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

        Delta_tt_cc, res = solver(lhs_matrix_tt, rhs_vec_tt, status.mals_delta0, status.kkt_iterations, status.is_last_iter, status.eta)
        status.mals_delta0 = Delta_tt_cc
        Delta_X_tt_cc = _tt_symmetrise(tt_reshape(_tt_get_block(1, Delta_tt_cc), (2, 2)), status.eps)
        Delta_Z_tt_cc = _tt_symmetrise(tt_reshape(_tt_get_block(2, Delta_tt_cc), (2, 2)), status.eps)
        Delta_Y_tt_cc = tt_rank_reduce(_tt_get_block(0, Delta_tt_cc), eps=status.eps)
        Delta_Y_tt_cc = tt_rank_reduce(tt_sub(Delta_Y_tt_cc, tt_fast_matrix_vec_mul(status.lag_map_y, Delta_Y_tt_cc, status.eps)), eps=status.eps)
        Delta_X_tt = tt_rank_reduce(tt_add(Delta_X_tt_cc, Delta_X_tt), eps=status.eps)
        Delta_Y_tt = tt_rank_reduce(tt_add(Delta_Y_tt_cc, Delta_Y_tt), eps=status.eps)
        Delta_Z_tt = tt_rank_reduce(tt_add(Delta_Z_tt_cc, Delta_Z_tt), eps=status.eps)
        if status.ineq_status is IneqStatus.ACTIVE:
            Delta_T_tt_cc = tt_rank_reduce(_tt_get_block(3, Delta_tt_cc), eps=status.eps)
            Delta_T_tt_cc = tt_fast_hadammard(ineq_mask, tt_reshape(Delta_T_tt_cc, (2, 2)), status.eps)
            Delta_T_tt = tt_rank_reduce(tt_add(Delta_T_tt_cc, Delta_T_tt), eps=status.eps)

        x_step_size, z_step_size = _tt_line_search(
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

    return x_step_size, z_step_size, Delta_X_tt, Delta_Y_tt, Delta_Z_tt, Delta_T_tt, status


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
        X_tt = tt_add(X_tt, tt_scale(status.boundary_val, tt_identity(len(X_tt))))
        Z_tt = tt_add(Z_tt, tt_scale(status.boundary_val, tt_identity(len(Z_tt))))

    x_step_size, status.eigen_x0 = tt_max_generalised_eigen(X_tt, Delta_X_tt, x0=status.eigen_x0, tol=status.eps, verbose=status.verbose)
    z_step_size, status.eigen_z0 = tt_max_generalised_eigen(Z_tt, Delta_Z_tt, x0=status.eigen_z0, tol=status.eps, verbose=status.verbose)
    if status.ineq_status is not IneqStatus.NOT_IN_USE:
        if status.is_last_iter:
            X_tt = tt_add(X_tt, tt_scale(status.boundary_val, ineq_mask))
            T_tt = tt_add(T_tt, tt_scale(status.boundary_val, ineq_mask))
        x_step_size, z_step_size = _tt_line_search_ineq(x_step_size, z_step_size, X_tt, T_tt, Delta_X_tt, Delta_T_tt, ineq_mask, status)
    tau_x = 0.9 + 0.05*min(x_step_size,  z_step_size) if x_step_size < 1 else 1.0
    tau_z = 0.9 + 0.05*min(x_step_size,  z_step_size) if z_step_size < 1 else 1.0
    return tau_x*x_step_size, tau_z*z_step_size


def _ineq_step_size(A_tt, Delta_tt, e_tt, status):
    sum_tt = tt_add(A_tt, Delta_tt)
    if status.compl_ineq_mask:
        sum_tt = tt_add(sum_tt, tt_scale(tt_entrywise_sum(sum_tt)/status.num_ineq_constraints, status.compl_ineq_mask))
    sum_tt = tt_rank_reduce(sum_tt, status.eps)
    e_tt, min_eval = tt_min_eig(tt_diag_op(sum_tt, status.eps), x0=e_tt, tol=status.eps, verbose=status.verbose)
    e_tt_sq = tt_reshape(tt_normalise(tt_fast_hadammard(e_tt, e_tt, status.eps)), (2, 2))
    min_A_val = tt_inner_prod(A_tt, e_tt_sq)
    min_Delta_val = tt_inner_prod(Delta_tt, e_tt_sq)
    if min_Delta_val >= 0:
        step_size = 1
    else:
        step_size = np.clip(-(1-status.op_tol)*min_A_val/min_Delta_val, a_min=0, a_max=1)
    return step_size, e_tt


def _tt_line_search_ineq(x_step_size, z_step_size, X_tt, T_tt, Delta_X_tt, Delta_T_tt, ineq_mask, status):

    if x_step_size > 0:
        masked_X_tt = tt_fast_hadammard(ineq_mask, X_tt, status.eps)
        masked_Delta_X_tt = tt_fast_hadammard(ineq_mask, Delta_X_tt, status.eps)
        x_ineq_step_size, status.eigen_xt0 = _ineq_step_size(
            tt_add(masked_X_tt, tt_scale(status.boundary_val, ineq_mask)),
            tt_scale(x_step_size, masked_Delta_X_tt),
            status.eigen_xt0,
            status
        )
        if not status.is_last_iter:
            if 1 - x_ineq_step_size < status.eps:
                if status.ineq_status is IneqStatus.ACTIVE:
                    status.ineq_status = IneqStatus.SETTING_INACTIVE
            else:
                if status.ineq_status is IneqStatus.INACTIVE:
                    status.ineq_status = IneqStatus.SETTING_ACTIVE
        x_step_size *= x_ineq_step_size

    if z_step_size > 0 and status.ineq_status is IneqStatus.ACTIVE:
        t_step_size, status.eigen_zt0 = _ineq_step_size(
            T_tt,
            tt_scale(z_step_size, Delta_T_tt),
            status.eigen_zt0,
        status
        )
        z_step_size *= t_step_size

    return x_step_size, z_step_size


def _update(x_step_size, z_step_size, X_tt, Z_tt, Delta_X_tt, Delta_Z_tt, status):
    if 0 < x_step_size < 1e-5 and 0 < z_step_size < 1e-5:
        status.is_last_iter = True
    elif Delta_X_tt is not None and Delta_Z_tt is not None:
        status.is_last_iter = status.is_last_iter or (tt_norm(Delta_X_tt) + tt_norm(Delta_Z_tt) < status.eps)
        if status.is_last_iter:
            X_tt = _tt_symmetrise(tt_add(X_tt, tt_scale(x_step_size, Delta_X_tt)), status.op_tol)
        else:
            X_tt = _tt_psd_symmetrise(tt_add(X_tt, tt_scale(x_step_size, Delta_X_tt)), status.op_tol)
        if status.is_last_iter:
            Z_tt = _tt_symmetrise(tt_add(Z_tt, tt_scale(z_step_size, Delta_Z_tt)), status.op_tol)
        else:
            Z_tt = _tt_psd_symmetrise(tt_add(Z_tt, tt_scale(z_step_size, Delta_Z_tt)), status.op_tol)

    return X_tt, Z_tt


def _initialise(ineq_mask, status, dim):
    X_tt = tt_identity(dim)
    Z_tt = tt_identity(dim)
    Y_tt = tt_reshape(tt_zero_matrix(dim), (4, ))
    T_tt = None

    if status.ineq_status is IneqStatus.ACTIVE:
        T_tt = tt_scale(status.eps, ineq_mask)

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

    boundary_val: float = 0.01
    sigma: float = 0.5
    num_ineq_constraints: float = 0
    lag_map_t: list = None
    lag_map_y: list = None
    compl_ineq_mask = None
    mals_delta0 = None
    eigen_x0 = None
    eigen_z0 = None
    eigen_xt0 = None
    eigen_zt0 = None
    kkt_iterations = 6
    centrl_error_normalisation: float = 1.0
    eta = 1e-3


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
    abs_tol=1e-3,
    eps=1e-12,
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
        1
    )
    lag_maps = {key: tt_rank_reduce(value, eps=eps) for key, value in lag_maps.items()}
    obj_tt = tt_rank_reduce(obj_tt, eps=eps)
    lin_op_tt = tt_rank_reduce(lin_op_tt, eps=eps)
    bias_tt = tt_rank_reduce(bias_tt, eps=eps)

    # Normalisation
    # We normalise the objective to the scale of the average constraint
    status.primal_error_normalisation = 1 + tt_norm(bias_tt)
    status.dual_error_normalisation = 1 + tt_norm(obj_tt)
    status.feasibility_tol = feasibility_tol + (op_tol/2) # account for error introduced in psd_rank_reduce

    lhs_skeleton = TTBlockMatrix()
    lhs_skeleton[1, 2] = tt_reshape(tt_identity(2 * dim), (4, 4))
    solver_ineq = lambda lhs, rhs, x0, nwsp, refinement, termination_tol: tt_restarted_block_als(
        lhs,
        rhs,
        rank_restriction=min(4*dim + 2*dim, 25),
        x0=x0,
        local_solver=_ipm_local_solver_ineq,
        op_tol=op_tol,
        termination_tol=termination_tol,
        num_restarts=3,
        inner_m=nwsp,
        refinement=refinement,
        verbose=verbose
    )
    solver_eq = lambda lhs, rhs, x0, nwsp, refinement, termination_tol: tt_restarted_block_als(
        lhs,
        rhs,
        rank_restriction=max(3*dim + 2*dim, 25),
        x0=x0,
        local_solver=_ipm_local_solver,
        op_tol=op_tol,
        termination_tol=termination_tol,
        num_restarts=3,
        inner_m=nwsp,
        refinement=refinement,
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

    X_tt, Y_tt, Z_tt, T_tt = _initialise(ineq_mask, status, dim)

    iteration = 0
    finishing_steps = max_refinement
    prev_primal_error = status.primal_error
    prev_dual_error = status.dual_error
    prev_centrality_error = status.centrality_error
    lhs = lhs_skeleton

    while finishing_steps > 0:
        status.eta = max(0.5*status.eta, 0.5*feasibility_tol)
        iteration += 1
        status.aho_direction = (iteration > warm_up)
        status.is_last_iter = status.is_last_iter or (max_iter - max_refinement < iteration)
        ZX = tt_inner_prod(Z_tt, X_tt)
        TX = tt_inner_prod(X_tt, T_tt) + status.boundary_val*tt_entrywise_sum(T_tt) if status.ineq_status is IneqStatus.ACTIVE else 0
        status.mu = np.divide(abs(ZX) + abs(TX), (2 ** dim + (status.ineq_status is IneqStatus.ACTIVE)*status.num_ineq_constraints))
        status.centrl_error_normalisation = 1 + abs(tt_inner_prod(obj_tt, tt_reshape(X_tt, (4, ))))
        status.centrality_error = status.mu / status.centrl_error_normalisation
        status.is_central = np.less(status.centrality_error, (1 + (status.ineq_status is IneqStatus.ACTIVE))*centrality_tol)

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
            print(f"\nResults of iteration {iteration - 1}:")
            print(f"Finishing up: {status.is_last_iter}")
            print(f"Inequality active: {str(status.ineq_status)}")
            print(
                f"Is central: {status.is_central}, Is primal feasible:  {status.is_primal_feasible}, Is dual feasible: {status.is_dual_feasible}")
            print(f"Using AHO-Direction: {status.aho_direction}")
            print(f"Centrality Error: {status.centrality_error}")
            print(f"Primal-Dual Error: {status.primal_error, status.dual_error}")
            print(f"Sigma: {status.sigma}")
            print(
                f"Ranks X_tt: {tt_ranks(X_tt)}, Z_tt: {tt_ranks(Z_tt)}, \n"
                f"      Y_tt: {tt_ranks(Y_tt)}, T_tt: {tt_ranks(T_tt) if T_tt else None} \n"
            )

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
        if verbose:
            print(f"--- Step {iteration} ---")
            print(f"Step sizes: {x_step_size}, {z_step_size}")


        X_tt, Z_tt = _update(x_step_size, z_step_size, X_tt, Z_tt, Delta_X_tt, Delta_Z_tt, status)

        # TODO: transpose without reshape
        if z_step_size > 1e-5:
            Y_tt = tt_reshape(_tt_symmetrise(tt_reshape(tt_add(Y_tt, tt_scale(z_step_size, Delta_Y_tt)), (2, 2)), op_tol), (4, ))

        if status.ineq_status is IneqStatus.ACTIVE:
            if z_step_size > 1e-5:
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

        if status.is_last_iter:
            if abs(ZX) + abs(TX) <= abs_tol and status.primal_error < abs_tol and status.dual_error < abs_tol:
                finishing_steps -= max_refinement
            else:
                finishing_steps -= 1
            status.kkt_iterations += (finishing_steps == 1)

        if (
                abs(prev_primal_error - status.primal_error) < 0.04*gap_tol
                and abs(prev_dual_error - status.dual_error) < 0.04*gap_tol
                and abs(prev_centrality_error - status.centrality_error) < 0.02*gap_tol
                and not status.is_last_iter
        ):
            if status.verbose:
                print(
                    "==================================\n Progress stalled!\n==================================")
            status.is_last_iter = True

        prev_primal_error = status.primal_error
        prev_dual_error = status.dual_error
        prev_centrality_error = status.centrality_error

        #print()
        #print(tt_norm(X_tt), tt_norm(Delta_X_tt))
        #print(tt_norm(Y_tt), tt_norm(Delta_Y_tt))
        #print(tt_norm(Z_tt), tt_norm(Delta_Z_tt))
        #print()

    print(f"---Terminated---")
    print(f"Converged in {iteration} iterations.")
    print(
        f"Ranks X_tt: {tt_ranks(X_tt)}, Z_tt: {tt_ranks(Z_tt)}, \n"
        f"      Y_tt: {tt_ranks(Y_tt)}, T_tt: {tt_ranks(T_tt) if T_tt else None} \n"
    )
    return X_tt, Y_tt, T_tt, Z_tt, {"num_iters": iteration}