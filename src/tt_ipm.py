import sys
import os

sys.path.append(os.getcwd() + '/../')

from src.tt_ops import *
from src.tt_amen import tt_block_als, cached_einsum, TTBlockMatrix, TTBlockVector
from src.tt_eigen import tt_max_generalised_eigen, tt_min_eig, tt_mat_mat_mul
from dataclasses import dataclass

def forward_backward_sub(L, b, overwrite_b=False):
    y = scip.linalg.solve_triangular(L, b, lower=True, check_finite=False, overwrite_b=overwrite_b)
    x = scip.linalg.solve_triangular(L.T, y, lower=False, check_finite=False, overwrite_b=True)
    return x


def _get_eq_mat_vec(XAX_k, block_A_k, XAX_kp1, x_shape, inv_I):
    x_element_shape = (x_shape[0], x_shape[2], x_shape[3])
    XAX_k_00 = XAX_k[0, 0]
    XAX_k_01 = XAX_k[0, 1]
    XAX_k_21 = XAX_k[2, 1]
    XAX_k_22 = XAX_k[2, 2]
    XAX_kp1_00 = XAX_kp1[0, 0]
    XAX_kp1_01 = XAX_kp1[0, 1]
    XAX_kp1_21 = XAX_kp1[2, 1]
    XAX_kp1_22 = XAX_kp1[2, 2]
    block_A_k_00 = block_A_k[0, 0]
    block_A_k_01 = block_A_k[0, 1]
    block_A_k_21 = block_A_k[2, 1]
    block_A_k_22 = block_A_k[2, 2]

    K_y = contract_expression('lsr,smnS,LSR,rnR->lmL', XAX_k_00.shape, block_A_k_00.shape,  XAX_kp1_00.shape, x_element_shape, optimize="greedy")
    mL = contract_expression('lsr,smnS,LSR,rnR->lmL', XAX_k_01.shape, block_A_k_01.shape,  XAX_kp1_01.shape, x_element_shape, optimize="greedy")
    L_Z = contract_expression('lsr,smnS,LSR,rnR->lmL', XAX_k_21.shape, block_A_k_21.shape, XAX_kp1_21.shape, x_element_shape, optimize="greedy")
    # lsr,smnS,LSR,rnR  == abr, bdnf, gfR, rnR -> adg
    L_XmL_adj = contract_expression(
        'abr, bdnf, gfR, lsr,smnS,LSR,lmL -> adg',
        XAX_k_22.shape, block_A_k_22.shape, XAX_kp1_22.shape,
        XAX_k_01.shape, block_A_k_01.shape, XAX_kp1_01.shape,
        x_element_shape,
        optimize = "greedy"
    )

    def mat_vec(x_core):
        x_core = x_core.reshape(2, x_shape[0], x_shape[2], x_shape[3])
        result = np.zeros_like(x_core)
        result[0] += K_y(XAX_k_00, block_A_k_00, XAX_kp1_00, x_core[0]).__iadd__(mL(XAX_k_01, block_A_k_01, XAX_kp1_01, x_core[1]))
        result[1] += L_Z(XAX_k_21, block_A_k_21, XAX_kp1_21, x_core[1]).__isub__(
            L_XmL_adj( XAX_k_22, block_A_k_22, XAX_kp1_22,XAX_k_01, block_A_k_01, XAX_kp1_01, x_core[0]).__imul__(inv_I)
        )
        return result.reshape(-1, 1)

    return mat_vec

def _ipm_local_solver(XAX_k, block_A_k, XAX_k1, Xb_k, block_b_k, Xb_k1, previous_solution, nrmsc, size_limit, rtol):
    x_shape = previous_solution.shape
    m = x_shape[0] * x_shape[2] * x_shape[3]
    rhs = np.zeros_like(previous_solution)
    rhs[:, 0] = cached_einsum('br,bmB,BR->rmR', Xb_k[0], nrmsc * block_b_k[0], Xb_k1[0]) if 0 in block_b_k else 0
    rhs[:, 1] = cached_einsum('br,bmB,BR->rmR', Xb_k[1], nrmsc * block_b_k[1], Xb_k1[1]) if 1 in block_b_k else 0
    rhs[:, 2] = cached_einsum('br,bmB,BR->rmR', Xb_k[2], nrmsc * block_b_k[2], Xb_k1[2]) if 2 in block_b_k else 0
    inv_I = np.divide(1, cached_einsum('lsr,smnS,LSR->lmL', XAX_k[1, 2], block_A_k[1, 2], XAX_k1[1, 2]))
    norm_rhs = np.linalg.norm(rhs)
    block_res_old = np.linalg.norm(block_A_k.block_local_product(XAX_k, XAX_k1, previous_solution) - rhs) / norm_rhs
    if block_res_old < rtol:
        return previous_solution, block_res_old, block_res_old, rhs, norm_rhs
    if m <= size_limit:
        try:
            L_L_Z = scip.linalg.cholesky(
                cached_einsum('lsr,smnS,LSR->lmLrnR', XAX_k[2, 1], block_A_k[2, 1], XAX_k1[2, 1]).reshape(m, m),
                check_finite=False, lower=True, overwrite_a=True
            )
            mR_p = rhs[:, 0].reshape(m, 1)
            mR_d = rhs[:, 1].reshape(m, 1)
            mR_c = rhs[:, 2].reshape(m, 1)
            L_X = cached_einsum('lsr,smnS,LSR->lmLrnR', XAX_k[2, 2], block_A_k[2, 2], XAX_k1[2, 2]).reshape(m, m)
            mL_eq = cached_einsum('lsr,smnS,LSR->lmLrnR', XAX_k[0, 1], block_A_k[0, 1], XAX_k1[0, 1]).reshape(m, m)
            A = mL_eq @ forward_backward_sub(L_L_Z, L_X * inv_I.reshape(1, -1), overwrite_b=True) @ mL_eq.T + cached_einsum('lsr,smnS,LSR->lmLrnR', XAX_k[0, 0], block_A_k[0, 0], XAX_k1[0, 0]).reshape(m, m)
            b = mR_p - mL_eq @ forward_backward_sub(L_L_Z, mR_c - (L_X * inv_I.reshape(1, -1)) @ mR_d, overwrite_b=True) - A @ np.transpose(previous_solution, (1, 0, 2, 3))[1].reshape(-1, 1)
            y = scip.linalg.solve(A, b, check_finite=False, overwrite_a=True, overwrite_b=True) + np.transpose(previous_solution, (1, 0, 2, 3))[1].reshape(-1, 1)
            z = inv_I.reshape(-1, 1) * (mR_d - mL_eq.T @ y)
            x = forward_backward_sub(L_L_Z, mR_c - L_X @ z, overwrite_b=True)
            solution_now = np.transpose(np.vstack((y, x, z)).reshape(x_shape[1], x_shape[0], x_shape[2], x_shape[3]), (1, 0, 2, 3))
        except:
            size_limit = 0

    if m > size_limit:
        mat_vec = _get_eq_mat_vec(XAX_k, block_A_k, XAX_k1, x_shape, inv_I)

        linear_op = scip.sparse.linalg.LinearOperator((2 * m, 2 * m), matvec=mat_vec)
        local_rhs = -linear_op(np.transpose(previous_solution[:, :2], (1, 0, 2, 3)).reshape(-1, 1)).reshape(2, x_shape[0], x_shape[2], x_shape[3])
        local_rhs[0] += rhs[:, 0]
        local_rhs[1] += rhs[:, 2] - cached_einsum('lsr,smnS,LSR,rnR->lmL', XAX_k[2, 2], block_A_k[2, 2], XAX_k1[2, 2], inv_I*rhs[:, 1])
        solution_now, info = scip.sparse.linalg.gmres(
            linear_op,
            local_rhs.reshape(-1, 1),
            rtol=1e-3*block_res_old,
            maxiter=25
        )
        solution_now = np.transpose(solution_now.reshape(2, x_shape[0], x_shape[2], x_shape[3]), (1, 0, 2, 3)).__iadd__(previous_solution[:, :2])
        z = inv_I * (rhs[:, 1] - cached_einsum('lsr,smnS,LSR,lmL->rnR', XAX_k[0, 1], block_A_k[0, 1], XAX_k1[0, 1], solution_now[:, 0]))
        solution_now = np.concatenate((solution_now, z.reshape(x_shape[0], 1, x_shape[2], x_shape[3])), axis=1)

    block_res_new = np.linalg.norm(block_A_k.block_local_product(XAX_k, XAX_k1, solution_now).__isub__(rhs)) / norm_rhs

    if block_res_old < block_res_new:
        solution_now = previous_solution

    return solution_now, block_res_old, min(block_res_new, block_res_old), rhs, norm_rhs


def _get_ineq_mat_vec(XAX_k, block_A_k, XAX_kp1, x_shape, inv_I):
    x_element_shape = (x_shape[0], x_shape[2], x_shape[3])
    XAX_k_00 = XAX_k[0, 0]
    XAX_k_01 = XAX_k[0, 1]
    XAX_k_21 = XAX_k[2, 1]
    XAX_k_22 = XAX_k[2, 2]
    XAX_k_31 = XAX_k[3, 1]
    XAX_k_33 = XAX_k[3, 3]
    XAX_kp1_00 = XAX_kp1[0, 0]
    XAX_kp1_01 = XAX_kp1[0, 1]
    XAX_kp1_21 = XAX_kp1[2, 1]
    XAX_kp1_22 = XAX_kp1[2, 2]
    XAX_kp1_31 = XAX_kp1[3, 1]
    XAX_kp1_33 = XAX_kp1[3, 3]
    block_A_k_00 = block_A_k[0, 0]
    block_A_k_01 = block_A_k[0, 1]
    block_A_k_21 = block_A_k[2, 1]
    block_A_k_22 = block_A_k[2, 2]
    block_A_k_31 = block_A_k[3, 1]
    block_A_k_33 = block_A_k[3, 3]
    K_y = contract_expression('lsr,smnS,LSR,rnR->lmL', XAX_k_00.shape, block_A_k_00.shape,  XAX_kp1_00.shape, x_element_shape, optimize="greedy")
    mL = contract_expression('lsr,smnS,LSR,rnR->lmL', XAX_k_01.shape, block_A_k_01.shape,  XAX_kp1_01.shape, x_element_shape, optimize="greedy")
    mL_adj = contract_expression('lsr,smnS,LSR,lmL->rnR', XAX_k_01.shape, block_A_k_01.shape, XAX_kp1_01.shape, x_element_shape, optimize="greedy")
    L_X = contract_expression('lsr,smnS,LSR,rnR->lmL', XAX_k_22.shape, block_A_k_22.shape, XAX_kp1_22.shape, x_element_shape, optimize="greedy")
    L_Z = contract_expression('lsr,smnS,LSR,rnR->lmL', XAX_k_21.shape, block_A_k_21.shape, XAX_kp1_21.shape, x_element_shape, optimize="greedy")
    T_op = contract_expression('lsr,smnS,LSR,rnR->lmL', XAX_k_31.shape, block_A_k_31.shape, XAX_kp1_31.shape, x_element_shape, optimize="greedy")
    K_t = contract_expression('lsr,smnS,LSR,rnR->lmL', XAX_k_33.shape, block_A_k_33.shape, XAX_kp1_33.shape, x_element_shape, optimize="greedy")

    def mat_vec(x_core):
        x_core = x_core.reshape(3, x_shape[0], x_shape[2], x_shape[3])
        result = np.zeros_like(x_core)
        result[0] += K_y(XAX_k_00, block_A_k_00, XAX_kp1_00, x_core[0]).__iadd__(mL(XAX_k_01, block_A_k_01, XAX_kp1_01, x_core[1]))
        result[1] += L_Z(XAX_k_21, block_A_k_21, XAX_kp1_21, x_core[1]).__isub__(
            L_X( XAX_k_22, block_A_k_22, XAX_kp1_22, (mL_adj(XAX_k_01, block_A_k_01, XAX_kp1_01, x_core[0]).__imul__(inv_I)).__iadd__(x_core[2]))
        )
        result[2] += T_op(XAX_k_31, block_A_k_31, XAX_kp1_31, x_core[1]).__iadd__(K_t(XAX_k_33, block_A_k_33, XAX_kp1_33, x_core[2]))
        return result.reshape(-1, 1)

    return mat_vec

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
    block_res_old_scalar = np.linalg.norm(block_A_k.block_local_product(XAX_k, XAX_k1, previous_solution) - rhs) / norm_rhs
    if block_res_old_scalar < rtol:
        return previous_solution, block_res_old_scalar, block_res_old_scalar, rhs, norm_rhs
    if m <= size_limit:
        try:
            L_L_Z = scip.linalg.cholesky(
                cached_einsum('lsr,smnS,LSR->lmLrnR', XAX_k[(2, 1)], block_A_k[(2, 1)], XAX_k1[(2, 1)]).reshape(m, m),
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
                    - A @ previous_solution[:, 1].reshape(-1, 1)
                    - B @ previous_solution[:, 3].reshape(-1, 1)
            )
            v = (
                    mR_t - T_op @ (L_L_Z_inv_mR_c - forward_backward_sub(L_L_Z, (L_X * inv_I.reshape(1, -1)) @ mR_d, overwrite_b=True))
                    - C @ previous_solution[:, 1].reshape(-1, 1)
                    - D @ previous_solution[:, 3].reshape(-1, 1)
            )
            Dlu, Dpiv = scip.linalg.lu_factor(D, check_finite=False, overwrite_a=True)
            rhs_l = u.__isub__(B @ scip.linalg.lu_solve((Dlu, Dpiv), v, check_finite=False))
            lhs_l = A.__isub__(B.__imatmul__(scip.linalg.lu_solve((Dlu, Dpiv), C, check_finite=False)))
            y = scip.linalg.lu_solve(scip.linalg.lu_factor(lhs_l, check_finite=False, overwrite_a=True), rhs_l, check_finite=False, overwrite_b=True)
            t = scip.linalg.lu_solve((Dlu, Dpiv), v.__isub__(C @ y), check_finite=False, overwrite_b=True)
            y += previous_solution[:, 1].reshape(-1, 1)
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
        mat_vec = _get_ineq_mat_vec(XAX_k, block_A_k, XAX_k1, x_shape, inv_I)
        linear_op = scip.sparse.linalg.LinearOperator((3 * m, 3 * m), matvec=mat_vec)
        local_rhs = -linear_op(np.transpose(previous_solution[:, [0, 1, 3]], (1, 0, 2, 3)).reshape(-1, 1)).reshape(3, x_shape[0], x_shape[2], x_shape[3])
        local_rhs[0] += rhs[:, 0]
        local_rhs[1] += rhs[:, 2] - cached_einsum('lsr,smnS,LSR,rnR->lmL', XAX_k[2, 2], block_A_k[2, 2],
                                                  XAX_k1[2, 2], inv_I * rhs[:, 1])
        local_rhs[2] += rhs[:, 3]
        solution_now, info = scip.sparse.linalg.gmres(
            linear_op,
            local_rhs.reshape(-1, 1),
            rtol=1e-3*block_res_old_scalar,
            maxiter=25,
        )
        solution_now = np.transpose(solution_now.reshape(3, x_shape[0], x_shape[2], x_shape[3]),
                                    (1, 0, 2, 3)) + previous_solution[:, [0, 1, 3]]
        z = inv_I * (
                    rhs[:, 1] - cached_einsum('lsr,smnS,LSR,lmL->rnR', XAX_k[0, 1], block_A_k[0, 1], XAX_k1[0, 1],
                                              solution_now[:, 0])) - solution_now[:, 2]
        solution_now = np.concatenate(
            (solution_now[:, :2], z.reshape(x_shape[0], 1, x_shape[2], x_shape[3]), solution_now[:, None, -1]), axis=1)

    block_res_new_scalar = np.linalg.norm(block_A_k.block_local_product(XAX_k, XAX_k1, solution_now) - rhs) / norm_rhs

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
    rhs = TTBlockVector()

    # Check primal feasibility and compute residual
    primal_feas = tt_rank_reduce(tt_sub(tt_fast_matrix_vec_mul(lin_op_tt, tt_reshape(X_tt, (4, )), status.eps), bias_tt), status.op_tol)  # primal feasibility
    status.primal_error = np.divide(tt_norm(primal_feas), status.primal_error_normalisation)
    status.is_primal_feasible = np.less(status.primal_error, status.feasibility_tol)

    # Check dual feasibility and compute residual
    dual_feas = tt_rank_reduce(tt_sub(tt_fast_matrix_vec_mul(lin_op_tt_adj, Y_tt, status.eps), tt_rank_reduce(tt_add(tt_reshape(Z_tt, (4, )), obj_tt), status.eps)), status.eps if status.with_ineq else status.op_tol)
    if status.with_ineq:
        dual_feas = tt_rank_reduce(tt_sub(dual_feas, tt_reshape(T_tt, (4, ))), status.op_tol)
    status.dual_error = np.divide(tt_norm(dual_feas), status.dual_error_normalisation)
    status.is_dual_feasible = np.less(status.dual_error, (1 + status.with_ineq)*status.feasibility_tol)

    status.is_last_iter = status.is_last_iter or (status.is_primal_feasible and status.is_dual_feasible and status.is_central)

    status.aho_direction = (2*status.centrality_error < max(status.primal_error, status.dual_error))

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
        if status.aho_direction:
            rhs[2] = tt_reshape(tt_scale(-1, _tt_symmetrise(tt_mat_mat_mul(X_tt, Z_tt, status.op_tol, status.eps), status.op_tol)), (4, ))
        else:
            rhs[2] = tt_reshape(tt_scale(-1, tt_mat_mat_mul(Z_tt, X_tt, status.op_tol, status.eps)), (4,))


    if status.with_ineq:
        lhs[3, 1] =  tt_diag_op(T_tt, status.op_tol)
        masked_X_tt = tt_rank_reduce(tt_add(tt_scale(status.boundary_val, ineq_mask), tt_fast_hadammard(ineq_mask, X_tt, status.eps)), eps=status.eps)
        lhs[3, 3] = tt_rank_reduce(tt_add(status.lag_map_t, tt_diag_op(masked_X_tt, status.eps)), eps=status.op_tol)
        if not status.is_central or status.is_last_iter:
            rhs[3] = tt_rank_reduce(tt_reshape(tt_scale(-1, tt_fast_hadammard(masked_X_tt, T_tt, status.eps)), (4, )), eps=status.op_tol)

    return lhs, rhs, status

def _tt_symmetrise(matrix_tt, err_bound):
    return tt_rank_reduce(tt_scale(0.5, tt_add(matrix_tt, tt_transpose(matrix_tt))), eps=err_bound)

def _tt_psd_symmetrise(matrix_tt, err_bound):
    return tt_psd_rank_reduce(tt_scale(0.5, tt_add(matrix_tt, tt_transpose(matrix_tt))), eps=err_bound)


def _tt_mask_symmetrise(matrix_tt, mask_tt, err_bound):
    return tt_mask_rank_reduce(tt_scale(0.5, tt_add(matrix_tt, tt_transpose(matrix_tt))), mask_tt, eps=err_bound)

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
        print("--- Predictor  step ---", flush=True)
    Delta_tt, res = solver(lhs_matrix_tt, rhs_vec_tt, status.mals_delta0, status.kkt_iterations, 0 if status.is_last_iter else None)
    status.mals_delta0 = Delta_tt
    if res < status.local_res_bound:
        Delta_X_tt = _tt_symmetrise(tt_reshape(_tt_get_block(1, Delta_tt), (2, 2)), status.eps)
        Delta_Z_tt = _tt_symmetrise(tt_reshape(_tt_get_block(2, Delta_tt), (2, 2)), status.eps)
        Delta_Y_tt = tt_rank_reduce(_tt_get_block(0, Delta_tt), eps=status.eps)
        Delta_Y_tt = tt_rank_reduce(tt_sub(Delta_Y_tt, tt_fast_matrix_vec_mul(status.lag_map_y, Delta_Y_tt, status.eps)), eps=status.eps)
        Delta_T_tt = None
        if status.with_ineq:
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
    else:
        dim = len(X_tt)
        status.kkt_iterations = min(status.kkt_iterations + 1, 20)
        if status.is_last_iter:
            Delta_X_tt = tt_zero_matrix(dim, (2, 2))
            Delta_Z_tt = tt_zero_matrix(dim, (2, 2))
            Delta_Y_tt = [np.zeros((1, 4, 1)) for _ in range(dim)]
            Delta_T_tt = None
            if status.with_ineq:
                Delta_T_tt = tt_zero_matrix(dim, (2, 2))
            x_step_size = 1
            z_step_size = 1
        else:
            if status.verbose:
                print("==================================\n Inaccurate results: Regularising!\n==================================")
            # regularise
            reg_tt = tt_scale(2*status.op_tol, tt_identity(dim))
            Delta_X_tt = reg_tt
            Delta_Z_tt = reg_tt
            Delta_Y_tt = [np.zeros((1, 4, 1)) for _ in range(dim)]
            Delta_T_tt = None
            if status.with_ineq:
                Delta_T_tt = tt_scale(2*status.eps, ineq_mask)
            x_step_size = 1
            z_step_size = 1
            status.primal_error += 2 * status.op_tol
            status.dual_error += 2 * status.op_tol
            status.centrality_error += 4 * status.op_tol
        return x_step_size, z_step_size, Delta_X_tt, Delta_Y_tt, Delta_Z_tt, Delta_T_tt, status

    if not status.is_central and not status.is_last_iter:

        DXZ = tt_inner_prod(Delta_X_tt, Delta_Z_tt)
        # Corrector
        if status.verbose:
            print(f"\n--- Centering-Corrector  step ---", flush=True)

        if status.with_ineq:
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
            """
            #Too expensive
            rhs_3 = tt_add(
                    tt_scale(status.sigma * status.mu, tt_reshape(ineq_mask, (4,))),
                    tt_sub(
                        rhs_vec_tt.get_row(3),
                        tt_reshape(tt_fast_hadammard(Delta_T_tt, Delta_X_tt, status.op_tol), (4,))
                    )
                ) if status.sigma > 0 else tt_sub(rhs_vec_tt.get_row(3), tt_reshape(tt_fast_hadammard(Delta_T_tt, Delta_X_tt, status.op_tol), (4,)))
            rhs_vec_tt[3] = tt_rank_reduce(
                rhs_3,
                status.op_tol
        )
        """
        else:
            mu_aff = (
                ZX + x_step_size * z_step_size * DXZ
                + z_step_size * tt_inner_prod(X_tt,Delta_Z_tt)
                + x_step_size * tt_inner_prod(Delta_X_tt, Z_tt)
            )
            e = max(1, 3*min(x_step_size, z_step_size)**2)
            status.sigma = min(0.99, (mu_aff/ZX) ** e)


        if DXZ > status.op_tol:
            if status.aho_direction:
                Delta_XZ_term = tt_scale(-1, _tt_symmetrise(tt_mat_mat_mul(Delta_X_tt, Delta_Z_tt, status.op_tol, status.eps, verbose=status.verbose), status.op_tol))
            else:
                Delta_XZ_term = tt_scale(-1, tt_mat_mat_mul(Delta_Z_tt, Delta_X_tt, status.op_tol, status.eps, verbose=status.verbose))
            rhs_vec_tt[2] = tt_rank_reduce(
                tt_add(
                    tt_scale(status.sigma * status.mu, tt_reshape(tt_identity(len(X_tt)), (4,))),
                    tt_add(
                        rhs_vec_tt.get_row(2),
                        tt_reshape(Delta_XZ_term, (4,))
                    )
                ),
                status.op_tol
            ) if status.sigma > 0 else tt_add(rhs_vec_tt.get_row(2), tt_reshape(Delta_XZ_term, (4,)))
        else:
            rhs_vec_tt[2] = tt_rank_reduce(
                tt_add(
                    tt_scale(status.sigma * status.mu, tt_reshape(tt_identity(len(X_tt)), (4,))),
                    rhs_vec_tt.get_row(2)
                ),
                status.op_tol
            ) if status.sigma > 0 else rhs_vec_tt.get_row(2)
        Delta_tt_cc, res = solver(lhs_matrix_tt, rhs_vec_tt, status.mals_delta0, status.kkt_iterations, 0 if status.is_last_iter else None)
        status.mals_delta0 = Delta_tt_cc
        if res < status.local_res_bound:
            Delta_X_tt_cc = _tt_symmetrise(tt_reshape(_tt_get_block(1, Delta_tt_cc), (2, 2)), status.eps)
            Delta_Z_tt_cc = _tt_symmetrise(tt_reshape(_tt_get_block(2, Delta_tt_cc), (2, 2)), status.eps)
            Delta_X_tt = tt_rank_reduce(tt_add(Delta_X_tt_cc, Delta_X_tt), eps=status.eps)
            Delta_Z_tt = tt_rank_reduce(tt_add(Delta_Z_tt_cc, Delta_Z_tt), eps=status.eps)
            if status.with_ineq:
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
            Delta_Y_tt_cc = tt_rank_reduce(_tt_get_block(0, Delta_tt_cc), eps=status.eps)
            Delta_Y_tt_cc = tt_rank_reduce(tt_sub(Delta_Y_tt_cc, tt_fast_matrix_vec_mul(status.lag_map_y, Delta_Y_tt_cc, status.eps)), eps=status.eps)
            Delta_Y_tt = tt_rank_reduce(tt_add(Delta_Y_tt_cc, Delta_Y_tt), eps=status.eps)
        else:
            dim = len(X_tt)
            status.kkt_iterations = min(status.kkt_iterations + 1, 20)
            if status.is_last_iter:
                Delta_X_tt = tt_zero_matrix(dim, (2, 2))
                Delta_Z_tt = tt_zero_matrix(dim, (2, 2))
                Delta_Y_tt = [np.zeros((1, 4, 1)) for _ in range(dim)]
                Delta_T_tt = None
                if status.with_ineq:
                    Delta_T_tt = tt_zero_matrix(dim, (2, 2))
                x_step_size = 1
                z_step_size = 1
            else:
                if status.verbose:
                    print("==================================\n Inaccurate results: Regularising!\n==================================")
                # regularise
                reg_tt = tt_scale(2 * status.op_tol, tt_identity(dim))
                Delta_X_tt = reg_tt
                Delta_Z_tt = reg_tt
                Delta_Y_tt = [np.zeros((1, 4, 1)) for _ in range(dim)]
                Delta_T_tt = None
                if status.with_ineq:
                    Delta_T_tt = tt_scale(2*status.eps, ineq_mask)
                x_step_size = 1
                z_step_size = 1
                status.primal_error += 2*status.op_tol
                status.dual_error += 2*status.op_tol
                status.centrality_error += 4*status.op_tol
            return x_step_size, z_step_size, Delta_X_tt, Delta_Y_tt, Delta_Z_tt, Delta_T_tt, status
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

    x_step_size, status.eigen_x0 = tt_max_generalised_eigen(X_tt, Delta_X_tt, x0=status.eigen_x0, tol=0.5*status.feasibility_tol, verbose=status.verbose)

    ############################
    #A = tt_matrix_to_matrix(Z_tt)
    #B = tt_matrix_to_matrix(Delta_Z_tt)
    #L_inv = np.linalg.inv(scip.linalg.cholesky(A, check_finite=False, lower=True))
    #eig_val, _ = scip.sparse.linalg.eigsh(-L_inv @ B @ L_inv.T, k=1, which="LA")
    #step_size = 1 / eig_val[0]
    #print("===================================")
    #print("Step size:", step_size)
    ##############################

    z_step_size, status.eigen_z0 = tt_max_generalised_eigen(Z_tt, Delta_Z_tt, x0=status.eigen_z0, tol=0.5*status.feasibility_tol, verbose=status.verbose)

    #print(z_step_size)
    #print()
    permitted_err_t = 1
    if status.with_ineq:
        if status.is_last_iter:
            X_tt = tt_add(X_tt, tt_scale(status.boundary_val, ineq_mask))
            T_tt = tt_add(T_tt, tt_scale(status.boundary_val, ineq_mask))
        x_step_size, z_step_size = _tt_line_search_ineq(x_step_size, z_step_size, X_tt, T_tt, Delta_X_tt, Delta_T_tt, ineq_mask, status)
    tau_x = 0.9 + 0.05*min(x_step_size,  z_step_size) if x_step_size < 1 else 1.0
    tau_z = 0.9 + 0.05*min(x_step_size,  z_step_size) if z_step_size < 1 else 1.0
    return tau_x*x_step_size, tau_z*z_step_size


def _ineq_step_size(A_tt, Delta_tt, status):
    sum_tt = tt_add(A_tt, Delta_tt)
    if status.compl_ineq_mask:
        sum_tt = tt_add(sum_tt, tt_scale(tt_entrywise_sum(sum_tt)/status.num_ineq_constraints, status.compl_ineq_mask))
    sum_tt = tt_rank_reduce(sum_tt, status.eps)
    e_tt, min_eval = tt_min_eig(tt_diag_op(sum_tt, status.eps), tol=0.5*status.feasibility_tol, verbose=status.verbose)
    e_tt = tt_reshape(tt_normalise(tt_fast_hadammard(e_tt, e_tt, status.eps)), (2, 2))
    min_A_val = tt_inner_prod(A_tt, e_tt)
    min_Delta_val = tt_inner_prod(Delta_tt, e_tt)
    if min_Delta_val >= 0:
        step_size = 1
    else:
        step_size = np.clip(-(1-status.op_tol)*min_A_val/min_Delta_val, a_min=0, a_max=1)
    return step_size


def _tt_line_search_ineq(x_step_size, z_step_size, X_tt, T_tt, Delta_X_tt, Delta_T_tt, ineq_mask, status):

    if x_step_size > 0:
        masked_X_tt = tt_fast_hadammard(ineq_mask, X_tt, status.eps)
        masked_Delta_X_tt = tt_fast_hadammard(ineq_mask, Delta_X_tt, status.eps)
        x_ineq_step_size = _ineq_step_size(
            tt_add(masked_X_tt, tt_scale(status.boundary_val, ineq_mask)),
            tt_scale(x_step_size, masked_Delta_X_tt),
            status
        )
        x_step_size *= x_ineq_step_size

    if z_step_size > 0:
        t_step_size = _ineq_step_size(
            T_tt,
            tt_scale(z_step_size, Delta_T_tt),
            status
        )
        z_step_size *= t_step_size

    return x_step_size, z_step_size


def _update(x_step_size, z_step_size, X_tt, Z_tt, Delta_X_tt, Delta_Z_tt, status):
    if x_step_size <1e-4 and z_step_size < 1e-4:
        status.is_last_iter = True
    else:
        if x_step_size > 1e-4:
            if status.is_last_iter:
                X_tt = _tt_symmetrise(tt_add(X_tt, tt_scale(x_step_size, Delta_X_tt)), status.op_tol)
            else:
                X_tt = _tt_psd_symmetrise(tt_add(X_tt, tt_scale(x_step_size, Delta_X_tt)), status.op_tol)
        elif z_step_size > 1e-4:
            X_tt = tt_add(X_tt, tt_scale(status.op_tol, tt_identity(len(X_tt))))
        if z_step_size > 1e-4:
            if status.is_last_iter:
                Z_tt = _tt_symmetrise(tt_add(Z_tt, tt_scale(z_step_size, Delta_Z_tt)), status.op_tol)
            else:
                Z_tt = _tt_psd_symmetrise(tt_add(Z_tt, tt_scale(z_step_size, Delta_Z_tt)), status.op_tol)
        elif x_step_size > 1e-4:
            Z_tt = tt_add(Z_tt, tt_scale(status.op_tol, tt_identity(len(Z_tt))))

    return X_tt, Z_tt

def _initialise(ineq_mask, status, dim):
    scale = 2**(dim/2)
    X_tt = tt_scale(scale, tt_identity(dim))
    Z_tt = tt_scale(scale, tt_identity(dim))
    Y_tt = tt_reshape(tt_zero_matrix(dim), (4, ))
    T_tt = None

    if status.with_ineq:
        T_tt = tt_scale(scale, ineq_mask)


    #print(tt_matrix_to_matrix(X_tt))
    #print(tt_matrix_to_matrix(Z_tt))
    #print(tt_matrix_to_matrix(tt_reshape(Y_tt, (2,  2))))
    #print(tt_matrix_to_matrix(T_tt))
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
    centrality_error: float
    mu: float
    is_last_iter: bool
    with_ineq: bool
    verbose: bool

    primal_error_normalisation: float
    dual_error_normalisation: float

    boundary_val: float = 0.01
    sigma: float = 0.5
    num_ineq_constraints: float = 0
    local_res_bound: float = 0.5
    lag_map_t: list = None
    lag_map_y: list = None
    compl_ineq_mask = None
    mals_delta0 = None
    eigen_x0 = None
    eigen_z0 = None
    kkt_iterations = 6


def tt_ipm(
    lag_maps,
    obj_tt,
    lin_op_tt,
    bias_tt,
    ineq_mask=None,
    max_iter=100,
    max_refinement=5,
    gap_tol=1e-4,
    aho_direction=True,
    op_tol=1e-5,
    abs_tol=1e-3,
    eps=1e-12,
    verbose=False,
):
    centrality_tol = gap_tol
    feasibility_tol = 2*gap_tol
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
        np.inf,
        False,
        ineq_mask is not None,
        verbose,
        1,
        1,
        local_res_bound=max(1e-3*2**dim, 0.5)
    )
    lhs_skeleton = TTBlockMatrix()
    lhs_skeleton[1, 2] = tt_reshape(tt_identity(2 * dim), (4, 4))
    if status.with_ineq:
        solver = lambda lhs, rhs, x0, nwsp, size_limit: tt_block_als(
            lhs,
            rhs,
            x0=x0,
            local_solver=_ipm_local_solver_ineq,
            tol=0.5 * min(feasibility_tol, centrality_tol),
            nswp=nwsp,
            size_limit=size_limit,
            verbose=verbose,
            amen=False
        )
        status.num_ineq_constraints = tt_inner_prod(ineq_mask, ineq_mask)
        status.compl_ineq_mask = tt_rank_reduce(tt_sub(tt_one_matrix(dim), ineq_mask), eps=eps)
        status.lag_map_t = lag_maps["t"]
        lhs_skeleton.add_alias((1, 2), (1, 3)) #[(1, 3)] = tt_reshape(tt_identity(2 * dim), (4, 4))
    else:
        solver = lambda lhs, rhs, x0, nwsp, size_limit: tt_block_als(
            lhs,
            rhs,
            x0=x0,
            local_solver=_ipm_local_solver,
            tol=0.5 * min(feasibility_tol, centrality_tol),
            nswp=nwsp,
            size_limit=size_limit,
            verbose=verbose,
            amen=False
        )
        status.num_ineq_constraints = 0

    lag_maps = {key: tt_rank_reduce(value, eps=eps) for key, value in lag_maps.items()}
    obj_tt = tt_rank_reduce(obj_tt, eps=eps)
    lin_op_tt = tt_rank_reduce(lin_op_tt, eps=eps)
    bias_tt = tt_rank_reduce(bias_tt, eps=eps)

    # Normalisation
    # We normalise the objective to the scale of the average constraint
    status.primal_error_normalisation = 1 + tt_norm(bias_tt)
    status.dual_error_normalisation = 1 + tt_norm(obj_tt)
    status.feasibility_tol = feasibility_tol + (op_tol/2) # account for error introduced in psd_rank_reduce

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

    while finishing_steps > 0:
        iteration += 1
        status.is_last_iter = status.is_last_iter or (max_iter <= iteration)
        ZX = tt_inner_prod(Z_tt, X_tt)
        TX = tt_inner_prod(X_tt, T_tt) + status.boundary_val*tt_entrywise_sum(T_tt) if status.with_ineq else 0
        status.mu = np.divide(ZX + TX, (2 ** dim + status.num_ineq_constraints))
        status.centrality_error = status.mu / (1 + abs(tt_inner_prod(obj_tt, tt_reshape(X_tt, (4, )))))
        status.is_central = np.less(status.centrality_error, (1 + status.with_ineq)*centrality_tol)

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
            print(f"Centrality Error: {status.centrality_error}")
            print(f"Primal-Dual Error: {status.primal_error, status.dual_error}")
            print(f"Sigma: {status.sigma}")
            print(
                f"Ranks X_tt: {tt_ranks(X_tt)}, Z_tt: {tt_ranks(Z_tt)}, \n"
                f"      Y_tt: {tt_ranks(Y_tt)}, T_tt: {tt_ranks(T_tt) if T_tt else None} \n"
            )
            print(f"--- Step {iteration} ---")

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
            print(f"Step sizes: {x_step_size}, {z_step_size}")


        X_tt, Z_tt = _update(x_step_size, z_step_size, X_tt, Z_tt, Delta_X_tt, Delta_Z_tt, status)

        # TODO: transpose without reshape
        Y_tt = tt_reshape(_tt_symmetrise(tt_reshape(tt_add(Y_tt, tt_scale(z_step_size, Delta_Y_tt)), (2, 2)), op_tol), (4, ))

        if status.with_ineq:
            if status.is_last_iter:
                T_tt = _tt_symmetrise(tt_add(T_tt, tt_scale(z_step_size, Delta_T_tt)), op_tol)
            else:
                T_tt = _tt_mask_symmetrise(tt_add(T_tt, tt_scale(z_step_size, Delta_T_tt)), ineq_mask, op_tol)

        if status.is_last_iter:
            if abs(ZX) + abs(TX) <= (1 + status.with_ineq)*abs_tol and status.primal_error < abs_tol and status.dual_error < abs_tol:
                finishing_steps -= max_refinement
            else:
                finishing_steps -= 1

        if (
                abs(prev_primal_error - status.primal_error) < 0.1*gap_tol
                and abs(prev_dual_error - status.dual_error) < 0.1*gap_tol
                and abs(prev_centrality_error - status.centrality_error) < 0.1*gap_tol
                and not status.is_last_iter
        ):
            if status.verbose:
                print(
                    "==================================\n Progress stalled!\n==================================")
            status.is_last_iter = True
        prev_primal_error = status.primal_error
        prev_dual_error = status.dual_error
        prev_centrality_error = status.centrality_error

    print(f"---Terminated---")
    print(f"Converged in {iteration} iterations.")
    print(
        f"Ranks X_tt: {tt_ranks(X_tt)}, Z_tt: {tt_ranks(Z_tt)}, \n"
        f"      Y_tt: {tt_ranks(Y_tt)}, T_tt: {tt_ranks(T_tt) if T_tt else None} \n"
    )
    return X_tt, Y_tt, T_tt, Z_tt, {"num_iters": iteration}