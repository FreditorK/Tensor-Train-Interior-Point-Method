import copy
import sys
import os
from traceback import print_tb

import numpy as np
import scipy.linalg
import scipy.optimize
import scipy.sparse.linalg

sys.path.append(os.getcwd() + '/../')

from src.tt_ops import *
from src.tt_ops import tt_rank_reduce
from src.tt_amen import tt_block_amen,  svd_solve_local_system
from src.tt_eig import tt_min_eig
from src.tt_ineq_check import tt_is_geq, tt_is_geq_zero, tt_is_psd


def vec(matrix):
    tensor = np.reshape(matrix, [2] * int(np.log2(np.prod(matrix.shape))))
    n = len(tensor.shape)
    axes = sum([list(t) for t in zip([i for i in range(n // 2)], [i for i in range(n // 2, n)])], [])
    tensor = np.transpose(tensor, axes=axes)
    return np.reshape(tensor, (-1, 1))

def mat(vector):
    dim = int(np.sqrt(len(vector)))
    tensor = np.reshape(vector, [2] * int(np.log2(np.prod(vector.shape))))
    n = len(tensor.shape)
    axes = [i for i in range(0, n, 2)] + [i for i in range(1, n, 2)]
    tensor = np.transpose(tensor, axes=axes)
    return np.reshape(tensor, (dim, dim))

def forward_backward_sub(L, b):
    y = scip.linalg.solve_triangular(L, b, lower=True, check_finite=False)
    x = scip.linalg.solve_triangular(L.T, y, lower=False, check_finite=False, overwrite_b=True)
    return x

def ipm_solve_system(lhs, rhs, local_auxs, num_blocks):
    k =  num_blocks - 1
    block_dim = lhs.shape[0] // num_blocks

    L_eq = -lhs[block_dim:2*block_dim, :block_dim]
    L_Z = lhs[k*block_dim:, :block_dim]
    L_Z_inv = scip.linalg.inv(L_Z)
    L_eq_adj = -lhs[:block_dim, block_dim:2*block_dim]
    #I = lhs[:block_dim, k*block_dim:]
    inv_I = np.diag(np.divide(1, np.diagonal(lhs[:block_dim, k*block_dim:])))
    L_X = lhs[k * block_dim:, k * block_dim:]
    R_d = -rhs[:block_dim]
    R_p = -rhs[block_dim:2*block_dim]
    R_c = -rhs[k * block_dim:]
    K_temp = L_Z_inv @ L_X
    K = K_temp @ inv_I
    k = L_Z_inv @ R_c
    KR_dmk = K @ R_d - k

    if num_blocks > 3:
        TL_ineq = -lhs[2 * block_dim:3 * block_dim, :block_dim]
        L_ineq_adj = -lhs[:block_dim, 2 * block_dim:3 * block_dim]
        R_ineq = lhs[2 * block_dim:3 * block_dim, 2 * block_dim:3 * block_dim]
        R_t = -rhs[2 * block_dim:3 * block_dim]
        A = L_eq @ K @ L_eq_adj
        D = R_ineq + TL_ineq @ K @ L_ineq_adj
        alpha = 0.05*(np.linalg.norm(A)/ np.linalg.norm(local_auxs["y"]))
        delta = 0.05*(np.linalg.norm(D) / np.linalg.norm(local_auxs["t"]))
        A += alpha*local_auxs["y"]
        B = L_eq @ K @ L_ineq_adj
        C = TL_ineq @ K @ L_eq_adj
        D += delta*local_auxs["t"]

        u = L_eq @ KR_dmk - R_p
        v = TL_ineq @ KR_dmk - R_t
        D_inv = scip.linalg.inv(D, check_finite=False)
        sol = scip.linalg.solve(A - B @ D_inv @ C, u - B @ (D_inv @ v), check_finite=False, assume_a="gen")
        t = D_inv @ (v - C @ sol)
        y = sol
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
    A = L_eq @ K @ L_eq_adj
    lhs = A + 0.05*(np.linalg.norm(A)/ np.linalg.norm(local_auxs["y"]))*local_auxs["y"]
    rhs = L_eq @ KR_dmk - R_p
    sol = np.linalg.solve(lhs, rhs)
    y = sol
    R_dmL_eq_adj_y = R_d - L_eq_adj @ y
    x = K @ R_dmL_eq_adj_y - k
    z = -inv_I @ R_dmL_eq_adj_y

    #print("---")
    #print(np.linalg.norm(- L_eq_adj @ y + I @ z + R_d))
    #print(np.linalg.norm(- L_eq @ x + R_p))
    #print(np.linalg.norm(L_Z @ x + L_X @ z + R_c))
    #print("---")
    return np.vstack((x, y, z))


def scaling_matrcies(matrix):
    lam, Q = np.linalg.eigh(matrix)
    root_matrix = Q @ np.diag(np.sqrt(lam)) @ Q.T
    root_matrix_inv = Q @ np.diag(1/np.sqrt(lam)) @ Q.T
    return root_matrix, root_matrix_inv


def preconditioned_scaling_matrices(matrix, k, eps=0.01):
    dim = matrix.shape[0]
    I = np.eye(dim)
    lam, Q = scipy.linalg.eigh(matrix)
    lam = 1/np.sqrt(lam)
    # X^(-1/2)
    tau = (1-eps)*np.max(lam[k:]) + eps*np.min(lam[k:])
    U = Q[:, :k] @ np.diag(np.sqrt(lam[:k] - tau))
    S_inv = scipy.linalg.inv(tau*I[:k, :k] + U.T @ U)
    P = tau*I + U @ U.T
    P_inv = np.divide(1, tau)*(I - U @ S_inv @ U.T)
    return P, P_inv


def tt_style_kron(matrix_1, matrix_2):
    shape = (matrix_1.shape[0]*matrix_2.shape[0], matrix_1.shape[1]*matrix_2.shape[1])
    tensor_1 = np.reshape(matrix_1, [2] * int(np.log2(np.prod(matrix_1.shape))))
    tensor_2 = np.reshape(matrix_2, [2] * int(np.log2(np.prod(matrix_2.shape))))
    n = 2*len(tensor_1.shape)
    axes = sum([list(t) for t in zip([i for i in range(n // 2)], [i for i in range(n // 2, n)])], [])
    return np.transpose(np.tensordot(tensor_1, tensor_2, axes=0), axes=axes).reshape(*shape)

def infeasible_newton_system(
        lhs_skeleton,
        vec_obj,
        X,
        vec_Y,
        Z,
        T,
        mat_lin_op,
        mat_lin_op_adj,
        vec_bias,
        mat_lin_op_ineq,
        mat_lin_op_ineq_adj,
        vec_bias_ineq,
        sigma,
        mu,
        tol,
        feasibility_tol,
        eps,
        active_ineq
):
    mu = max(sigma * mu,  1e-3)
    idx_add = int(active_ineq)
    I = np.eye(len(Z))
    L_Z = tt_style_kron(Z, I) + tt_style_kron(I, Z)
    L_X = tt_style_kron(I, X) +  tt_style_kron(X, I)
    #P_inv, P = preconditioned_scaling_matrices(X, k=1) #scaling_matrcies(X)
    #L_Z = tt_style_kron((Z @ P_inv).T, P) + tt_style_kron(P.T, P_inv @ Z)
    #L_X = tt_style_kron(P_inv.T, P @ X) +  tt_style_kron((X @ P).T, P_inv)
    block_dim = len(L_Z)
    vec_X = vec(X)

    if active_ineq:
        ineq_res = vec_bias_ineq - mat_lin_op_ineq @ vec_X
        mat_ineq_res_op = np.diag(ineq_res.flatten())
        mat_T_op = np.diag(vec(T).flatten())
        lhs_skeleton[2*block_dim:3*block_dim, 2*block_dim:3*block_dim] = mat_ineq_res_op
        lhs_skeleton[2*block_dim:3*block_dim, :block_dim] = -mat_T_op @ mat_lin_op_ineq
    lhs_skeleton[(2 + idx_add)*block_dim:, :block_dim] = L_Z
    lhs_skeleton[(2 + idx_add)*block_dim:, (2 + idx_add)*block_dim:] = L_X

    rhs = np.zeros((len(lhs_skeleton), 1))
    dual_feas = mat_lin_op_adj @ vec_Y - (vec(Z) + vec_obj)
    primal_feas = mat_lin_op @ vec_X - vec_bias  # primal feasibility
    primal_error = np.trace(primal_feas.T @ primal_feas)
    #if primal_error > feasibility_tol:
    rhs[block_dim:2*block_dim] = primal_feas

    if active_ineq:
        vec_T = vec(T)
        dual_feas = dual_feas + mat_lin_op_ineq_adj @ vec_T
        # TODO: Does mu 1 not also be under mat_lin_op_tt_ineq, need to adjust mu 1 to have zeros where L(X) has zeros
        one = mat_lin_op_ineq_adj @ np.ones_like(vec_X)
        nu = max(sigma*np.sum(vec_T.T @ ineq_res)/T.shape[0], 0)
        primal_feas_ineq = nu*one -vec_T*ineq_res
        primal_ineq_error = np.trace(primal_feas_ineq.T @ primal_feas_ineq)
        #if primal_ineq_error > tol:
        rhs[2*block_dim:3*block_dim] = primal_feas_ineq
        #primal_error += primal_ineq_error

    dual_error = np.trace(dual_feas.T @ dual_feas)
    XZ_term = L_Z @ vec_X
    rhs[(2 + idx_add)*block_dim:] = 2*mu*vec(np.eye(len(X))) - XZ_term
    #if dual_error > feasibility_tol:
    rhs[:block_dim] = dual_feas

    return lhs_skeleton, rhs, primal_error + dual_error


def _symmetrise(matrix):
    return (matrix.T + matrix)/2


def _ipm_newton_step(
        lag_maps,
        vec_obj,
        lhs_skeleton,
        mat_lin_op,
        mat_lin_op_adj,
        vec_bias,
        mat_lin_op_ineq,
        mat_lin_op_ineq_adj,
        vec_bias_ineq,
        X,
        vec_Y,
        T,
        Z,
        tol,
        feasibility_tol,
        active_ineq,
        verbose,
        eps,
        sigma
):
    mu = np.trace(Z.T @ X)/(Z.shape[0])
    lhs_matrix, rhs_vec, primal_dual_error = infeasible_newton_system(
        lhs_skeleton,
        vec_obj,
        X,
        vec_Y,
        Z,
        T,
        mat_lin_op,
        mat_lin_op_adj,
        vec_bias,
        mat_lin_op_ineq,
        mat_lin_op_ineq_adj,
        vec_bias_ineq,
        sigma,
        mu,
        tol,
        feasibility_tol,
        eps,
        active_ineq
    )
    idx_add = int(active_ineq)
    num_blocks = 4 if active_ineq else 3
    Delta = ipm_solve_system(lhs_matrix, rhs_vec, lag_maps, num_blocks)
    block_dim = len(Delta) // num_blocks
    vec_Delta_Y = Delta[block_dim:2*block_dim]
    Delta_T = mat(Delta[2*block_dim:3*block_dim]) if active_ineq else None
    Delta_X = mat(Delta[:block_dim])
    Delta_Z = mat(Delta[(2 + idx_add)*block_dim:])
    Delta_X = _symmetrise(Delta_X)
    Delta_Z = _symmetrise(Delta_Z)
    x_step_size, z_step_size = _line_search(X, T, Z, Delta_X, Delta_T, Delta_Z, mat_lin_op_ineq, vec_bias_ineq, active_ineq)
    X = X + 0.98 * x_step_size*Delta_X
    vec_Y = vec_Y + 0.98 * z_step_size*vec_Delta_Y
    Z = Z + 0.98 * z_step_size*Delta_Z
    if active_ineq:
        # FIXME: Note that T_tt should grow large on the zeros of b - L_ineq(X_tt)
        T = T + 0.98 * z_step_size*Delta_T

    if verbose:
        print(f"Step sizes: {x_step_size}, {z_step_size}")

    #print("Report ---")
    #print("Y")
    #print(np.round(mat(vec_Y), decimals=3))
    #if active_ineq:
        #print("Delta T")
        #print(np.round(Delta_T, decimals=3))
    #    print("T")
    #    print(np.round(T, decimals=3))
    #print("Delta X")
    #print(np.round(Delta_X, decimals=3))
    #print("X")
    #print(np.round(X, decimals=3))
    #print("Delta Z")
    #print(np.round(Delta_Z, decimals=3))

    return X, vec_Y, T, Z, primal_dual_error, mu


def _line_search(
        X,
        T,
        Z,
        Delta_X,
        Delta_T,
        Delta_Z,
        lin_op_ineq,
        vec_bias_ineq,
        active_ineq,
        iters=15,
        eps=1e-10
):
    x_step_size = 1
    z_step_size = 1
    discount = 0.5
    discount_x = False
    discount_z = False

    for iter in range(iters):
        discount_x = np.min(np.linalg.eigvalsh(X + x_step_size * Delta_X)) >= 0
        if discount_x:
            break
        else:
            x_step_size *= discount
    if active_ineq and discount_x:
        for iter in range(iters):
            discount_x = np.all(vec_bias_ineq - lin_op_ineq @ vec(X + x_step_size * Delta_X) > -eps)
            if discount_x:
                break
            else:
                x_step_size *= discount

    for iter in range(iters):
        discount_z = np.min(np.linalg.eigvalsh(Z + z_step_size * Delta_Z)) >= 0
        if discount_z:
            break
        else:
            z_step_size *= discount
    if active_ineq and discount_z:
        for iter in range(iters):
            discount_z = np.min(T + z_step_size*Delta_T) > -eps
            if discount_z:
                break
            else:
                z_step_size *= discount
    return discount_x*x_step_size, discount_z*z_step_size



def ipm(
    lag_maps,
    obj_tt,
    lin_op_tt,
    bias_tt,
    lin_op_tt_ineq=None,
    bias_tt_ineq=None,
    max_iter=100,
    feasibility_tol=1e-5,
    centrality_tol=1e-2,
    verbose=False,
    eps=1e-10
):
    dim = len(obj_tt)
    active_ineq = lin_op_tt_ineq is not None or bias_tt_ineq is not None
    # Normalisation
    obj_tt = tt_normalise(obj_tt)
    factor = np.divide(1, np.sqrt(tt_inner_prod(lin_op_tt, lin_op_tt)))
    lin_op_tt = tt_scale(factor, lin_op_tt)
    bias_tt = tt_scale(factor, bias_tt)
    if active_ineq:
        factor = np.divide(1, np.sqrt(tt_inner_prod(lin_op_tt_ineq, lin_op_tt_ineq)))
        lin_op_tt_ineq = tt_scale(factor, lin_op_tt_ineq)
        bias_tt_ineq = tt_scale(factor, bias_tt_ineq)
    # -------------
    feasibility_tol = feasibility_tol / np.sqrt(dim)
    centrality_tol = centrality_tol / np.sqrt(dim)
    op_tol = 0.5*min(feasibility_tol, centrality_tol)
    lag_maps = {key: tt_matrix_to_matrix(value) for key, value in lag_maps.items()}
    vec_obj = vec(tt_matrix_to_matrix(obj_tt))
    lin_op = tt_matrix_to_matrix(lin_op_tt)
    vec_bias = vec(tt_matrix_to_matrix(bias_tt))

    block_size = len(vec_obj)
    lhs_skeleton = np.zeros(((3+int(active_ineq))*block_size, (3+int(active_ineq))*block_size))
    lin_op_adj = lin_op.T
    lhs_skeleton[:block_size, block_size:2*block_size] = -lin_op_adj
    lhs_skeleton[block_size:2*block_size, :block_size] = -lin_op
    lin_op_ineq = None
    lin_op_ineq_adj = None
    vec_bias_ineq = None
    if active_ineq:
        bias_ineq = tt_matrix_to_matrix(bias_tt_ineq)
        lin_op_ineq = tt_matrix_to_matrix(lin_op_tt_ineq)
        lin_op_ineq_adj = -lin_op_ineq.T
        lhs_skeleton[:block_size, 2*block_size:3*block_size] = -lin_op_ineq_adj
        lhs_skeleton[:block_size, 3*block_size:4*block_size] = np.eye(2**(2*dim))
        vec_bias_ineq = vec(bias_ineq)
    else:
        lhs_skeleton[:block_size, 2*block_size:3*block_size] = np.eye(2**(2*dim))
    X = np.eye(2**dim)
    vec_Y = vec(np.zeros((2**dim, 2**dim)))
    T = np.ones((2**dim, 2**dim))
    if active_ineq:
        T = mat(lin_op_ineq_adj @ vec(T))
    Z = np.eye(2**dim)
    iter = 0
    sigma = 0.5
    for iter in range(1, max_iter):
        X, vec_Y, T, Z, pd_error, mu = _ipm_newton_step(
            lag_maps,
            vec_obj,
            lhs_skeleton,
            lin_op,
            lin_op_adj,
            vec_bias,
            lin_op_ineq,
            lin_op_ineq_adj,
            vec_bias_ineq,
            X,
            vec_Y,
            T,
            Z,
            op_tol,
            feasibility_tol,
            active_ineq,
            verbose,
            eps,
            sigma
        )
        sigma *= max(min((pd_error / mu)**2, 1), 0.01)
        if verbose:
            print(f"---Step {iter}---")
            print(f"Duality Gap: {100 * np.abs(mu):.4f}%")
            print(f"Primal-Dual error: {pd_error:.8f}")
            print(f"Sigma: {sigma:.4f}")

        if np.less(pd_error, feasibility_tol) and np.less(np.abs(mu), centrality_tol):
                break
    if verbose:
        print(f"---Terminated---")
        print(f"Converged in {iter} iterations.")
    return X, mat(vec_Y), T, Z
