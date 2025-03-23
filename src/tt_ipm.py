import copy
import sys
import os

sys.path.append(os.getcwd() + '/../')

from src.tt_ops import *
from src.tt_ops import tt_rank_reduce
from src.tt_amen import tt_block_gmres
from src.tt_ineq_check import tt_pd_optimal_step_size


def forward_backward_sub(L, b):
    y = scip.linalg.solve_triangular(L, b, lower=True, check_finite=False)
    x = scip.linalg.solve_triangular(L.T, y, lower=False, check_finite=False, overwrite_b=True)
    return x

def ipm_solve_local_system(prev_sol, lhs, rhs, local_auxs, num_blocks, eps):
    k =  num_blocks - 1
    L_Z = lhs[(k, 0)]
    block_dim = L_Z.shape[-1]
    prev_y = prev_sol[block_dim:2*block_dim]

    L_L_Z = scip.linalg.cholesky(L_Z, check_finite=False, overwrite_a=True, lower=True)
    #L_Z_inv = np.linalg.inv(L_Z)
    I = lhs[(0, k)]
    inv_I = np.divide(1, np.diagonal(I)).reshape(-1, 1) # Don't produce diagonal matrix to save memory
    L_X = lhs[(k, k)]
    dual_nonzero = 0 in rhs
    primal_nonzero = 1 in rhs
    R_d = rhs[0] if dual_nonzero else 0
    R_p = rhs[1] if primal_nonzero else 0
    K = forward_backward_sub(L_L_Z, L_X)  * inv_I.reshape(1, -1) #L_Z_inv @ L_X
    k = -forward_backward_sub(L_L_Z, rhs[k]) # L_Z_inv @ R_c
    KR_dmk = - (K @ R_d + k) if dual_nonzero else -k

    if num_blocks > 3:
        prev_t = prev_sol[2*block_dim:3*block_dim]
        TL_ineq = -lhs[(2, 0)]
        L_ineq_adj = -lhs[(0, 2)]
        R_ineq = lhs[(2, 2)]
        R_t = -rhs[2]
        A = lhs[(1, 0)] @ K @ lhs[(0, 1)]
        D = R_ineq + TL_ineq @ K @ L_ineq_adj
        alpha = 0.5*(np.linalg.norm(A)/ np.linalg.norm(local_auxs["y"]))
        delta = 0.5*(np.linalg.norm(D) / np.linalg.norm(local_auxs["t"]))
        A += alpha*local_auxs["y"]
        B = -lhs[(1, 0)] @ K @ L_ineq_adj
        C = TL_ineq @ K @ (-lhs[(0, 1)])
        D += delta*local_auxs["t"]

        u = -lhs[(1, 0)] @ KR_dmk + R_p - A @ prev_y - B @ prev_t
        v = TL_ineq @ KR_dmk - R_t - C @ prev_y - D @ prev_t
        D_inv = scip.linalg.inv(D, check_finite=False)
        sol = scip.linalg.solve(A - B @ D_inv @ C, u - B @ (D_inv @ v), check_finite=False, assume_a="gen")
        t = D_inv @ (v - C @ sol) + prev_t
        y = sol + prev_y
        R_dmL_eq_adj_yt = lhs[(0, 1)] @ y - L_ineq_adj @ t - R_d
        x = K @ R_dmL_eq_adj_yt - k
        z = -inv_I * R_dmL_eq_adj_yt
        return np.vstack((x, y, t, z))

    A = lhs[(1, 0)] @ K @ lhs[(0, 1)]
    A = A + 0.5*(np.linalg.norm(A)/ np.linalg.norm(local_auxs["y"]))*local_auxs["y"]
    b = -lhs[(1, 0)] @ KR_dmk + R_p - A @ prev_y
    sol = scip.linalg.solve(A, b, overwrite_a=True, overwrite_b=True, check_finite=False)
    y = sol + prev_y
    R_dmL_eq_adj_y = lhs[(0, 1)] @ y - R_d
    x = K @ R_dmL_eq_adj_y - k
    z = -inv_I * R_dmL_eq_adj_y
    return np.vstack((x, y, z))

def ipm_error(lhs, sol, rhs, num_blocks):
    k = num_blocks - 1
    dual_nonzero = 0 in rhs
    primal_nonzero = 1 in rhs
    R_d = rhs[0] if dual_nonzero else 0
    R_p = rhs[1] if primal_nonzero else 0
    R_c = rhs[k]
    L_Z = lhs[(k, 0)]
    L_X = lhs[(k, k)]
    I = lhs[(0, k)]
    block_dim = L_Z.shape[-1]
    x = sol[:block_dim]
    y = sol[block_dim:2 * block_dim]
    z = sol[k * block_dim:]
    return np.array([np.linalg.norm(lhs[(0, 1)] @ y + I @ z - R_d), np.linalg.norm(lhs[(1, 0)] @ x - R_p), np.linalg.norm(L_Z @ x + L_X @ z - R_c)])


def tt_scaling_matrices(X_tt):
    dim = len(X_tt)
    rank_one_X_tt = tt_rank_retraction(tt_diag(tt_diagonal(copy.deepcopy(X_tt))), [1] * (dim - 1))
    root_X_tt = [np.diag(np.sqrt(np.diagonal(np.abs(c.squeeze())))).reshape(1, 2, 2, 1) for c in rank_one_X_tt]
    root_X_tt_inv = [np.diag(1./np.sqrt(np.diagonal(np.abs(c.squeeze())))).reshape(1, 2, 2, 1) for c in rank_one_X_tt]
    return root_X_tt, root_X_tt_inv

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
        L_Z = tt_rank_reduce(tt_add(tt_kron(P, Z_tt), tt_kron(Z_tt, P)), eps=op_tol, rank_weighted_error=True)
        L_X = tt_rank_reduce(tt_add(tt_kron(X_tt, P), tt_kron(P, X_tt)), eps=op_tol, rank_weighted_error=True)

    X_tt = tt_reshape(X_tt, (4, ))
    Y_tt = tt_reshape(Y_tt, (4, ))
    Z_tt = tt_reshape(Z_tt, (4, ))

    if active_ineq:
        ineq_res_tt = tt_sub(bias_tt_ineq, tt_fast_matrix_vec_mul(lin_op_tt_ineq, X_tt, eps))
        mat_ineq_res_op_tt = tt_diag(ineq_res_tt)
        lhs_skeleton[(2, 2)] = tt_rank_reduce(mat_ineq_res_op_tt, op_tol, rank_weighted_error=True)
        Tmat_lin_op_tt_ineq =  tt_fast_mat_mat_mul(tt_reshape(tt_diag(tt_vec(T_tt)), (4, 4)), lin_op_tt_ineq, eps)
        lhs_skeleton[(2, 0)] = tt_rank_reduce(tt_scale(-1, Tmat_lin_op_tt_ineq), eps=op_tol, rank_weighted_error=True)
    lhs_skeleton[(2 + idx_add, 0)] = L_Z
    lhs_skeleton[(2 + idx_add, 2 + idx_add)] = L_X

    rhs = {}
    dual_feas = tt_sub(tt_fast_matrix_vec_mul(lin_op_tt_adj, Y_tt, eps), tt_add(Z_tt, obj_tt))
    primal_feas = tt_rank_reduce(tt_sub(tt_fast_matrix_vec_mul(lin_op_tt, X_tt, eps), bias_tt), op_tol, rank_weighted_error=True)  # primal feasibility
    primal_error = tt_inner_prod(primal_feas, primal_feas)
    if primal_error > feasibility_tol:
        rhs[1] = primal_feas

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
    if dual_error > feasibility_tol:
        rhs[0] = dual_feas

    XZ_term = tt_fast_matrix_vec_mul(L_X, Z_tt, eps)
    rhs[2 + idx_add] = tt_rank_reduce(tt_scale(-1, XZ_term), op_tol, rank_weighted_error=True) # tt_rank_reduce(tt_sub(tt_scale(mu_mul*mu, tt_reshape(tt_identity(len(X_tt)), (4, ))), XZ_term), op_tol, rank_weighted_error=True)

    return lhs_skeleton, rhs, primal_error + dual_error


def _tt_symmetrise(matrix_tt, err_bound):
    return tt_rank_reduce(tt_scale(0.5, tt_add(matrix_tt, tt_transpose(matrix_tt))), eps=err_bound, rank_weighted_error=True)


def _tt_get_block(i, block_matrix_tt):
    return  block_matrix_tt[:-1] + [block_matrix_tt[-1][:, i]]

def _tt_ipm_newton_step(
            lag_maps,
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
            eps,
            active_ineq,
            solver,
            direction
):
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
        active_ineq,
        direction,
        eps
    )
    # Predictor
    print("--- Predictor  step ---")
    idx_add = int(active_ineq)
    Delta_tt, res = solver(lhs_matrix_tt, rhs_vec_tt, None, 4)
    Delta_X_tt = tt_rank_reduce(tt_reshape(_tt_get_block(0, Delta_tt), (2, 2)), eps=op_tol, rank_weighted_error=True)
    Delta_Z_tt = tt_rank_reduce(tt_reshape(_tt_get_block(2 + idx_add, Delta_tt), (2, 2)), eps=op_tol, rank_weighted_error=True)
    Delta_X_tt = _tt_symmetrise(Delta_X_tt, op_tol)
    Delta_Z_tt = _tt_symmetrise(Delta_Z_tt, op_tol)

    x_step_size, z_step_size, _ = _tt_line_search(
        X_tt, Z_tt,
        Delta_X_tt, Delta_Z_tt,
        op_tol, eps
    )

    #Corrector
    print("\n--- Centering-Corrector  step ---")
    dim = len(Z_tt)
    P = tt_identity(dim)
    if direction == "XZ":
        L_Z = tt_scale(-1, tt_rank_reduce(tt_kron(Delta_Z_tt, P), eps=op_tol, rank_weighted_error=True))
        mu_mul = 1
        # No rank increase for operators
    else:
        L_Z = tt_scale(-1, tt_rank_reduce(tt_add(tt_kron(P, Delta_Z_tt), tt_kron(Delta_Z_tt, P)), eps=op_tol, rank_weighted_error=True))
        mu_mul = 2

    ZX = tt_inner_prod(Z_tt, X_tt)
    sigma = ((ZX + x_step_size*z_step_size*tt_inner_prod(Delta_X_tt, Delta_Z_tt) + z_step_size*tt_inner_prod(X_tt, Delta_Z_tt) + x_step_size*tt_inner_prod(Delta_X_tt, Z_tt))/ZX)**3
    mu = np.divide(ZX, 2**dim)
    rhs_vec_tt[2 + idx_add] = tt_rank_reduce(
        tt_add(
            tt_scale(mu_mul*sigma*mu, tt_reshape(tt_identity(len(X_tt)), (4, ))),
            tt_add(
            rhs_vec_tt[2 + idx_add],
            tt_fast_matrix_vec_mul(L_Z, tt_reshape(Delta_X_tt, (4, )), eps)
            )
        ),
        op_tol,
        rank_weighted_error=True
    )
    Delta_tt, res = solver(lhs_matrix_tt, rhs_vec_tt, Delta_tt, 3)
    Delta_X_tt = tt_rank_reduce(tt_reshape(_tt_get_block(0, Delta_tt), (2, 2)), eps=op_tol, rank_weighted_error=True)
    Delta_Y_tt = tt_rank_reduce(tt_reshape(_tt_get_block(1, Delta_tt), (2, 2)), eps=op_tol, rank_weighted_error=True)
    Delta_Z_tt = tt_rank_reduce(tt_reshape(_tt_get_block(2 + idx_add, Delta_tt), (2, 2)), eps=op_tol, rank_weighted_error=True)
    Delta_X_tt = _tt_symmetrise(Delta_X_tt, op_tol)
    Delta_Z_tt = _tt_symmetrise(Delta_Z_tt, op_tol)

    x_step_size, z_step_size, _ = _tt_line_search(
        X_tt, Z_tt,
        Delta_X_tt, Delta_Z_tt,
        op_tol, eps
    )

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
    tau = 0.9 + 0.09*min(x_step_size, z_step_size)
    return tau*x_step_size, tau*z_step_size, min(permitted_err_x, permitted_err_z)


def tt_ipm(
    lag_maps,
    obj_tt,
    lin_op_tt,
    bias_tt,
    lin_op_tt_ineq=None,
    bias_tt_ineq=None,
    max_iter=100,
    feasibility_tol=1e-5,
    centrality_tol=1e-2,
    op_tol=1e-5,
    verbose=False,
    direction = "XZ",
    eps=1e-12
):
    active_ineq = lin_op_tt_ineq is not None or bias_tt_ineq is not None
    dim = len(obj_tt)
    Rmax = max(int(np.floor(2**(dim-1)*np.sqrt(1/dim) - 1)), 4*dim)
    # Normalisation
    factor = np.divide(1, np.sqrt(tt_inner_prod(obj_tt, obj_tt)))
    obj_tt = tt_scale(factor, obj_tt)
    factor = np.divide(1, np.sqrt(tt_inner_prod(lin_op_tt, lin_op_tt)))
    lin_op_tt = tt_scale(factor, lin_op_tt)
    bias_tt = tt_scale(factor, bias_tt)
    if active_ineq:
        factor = np.divide(2, np.sqrt(tt_inner_prod(lin_op_tt_ineq, lin_op_tt_ineq)))
        lin_op_tt_ineq = tt_scale(factor, lin_op_tt_ineq)
        bias_tt_ineq = tt_scale(factor, bias_tt_ineq)
    # -------------
    # Reshaping
    lag_maps = {key: tt_rank_reduce(tt_reshape(value, (4, 4)), eps=op_tol) for key, value in lag_maps.items()}
    obj_tt = tt_rank_reduce(tt_reshape(obj_tt, (4, )), eps=op_tol)
    lin_op_tt = tt_rank_reduce(tt_reshape(lin_op_tt, (4, 4)), eps=op_tol)
    bias_tt = tt_rank_reduce(tt_reshape(bias_tt, (4, )), eps=op_tol)
    # -------------

    num_blocks = 4 if active_ineq else 3
    solver = lambda lhs, rhs, x0, nwsp: tt_block_gmres(
        lhs,
        rhs,
        x0=x0,
        tols=0.1 * np.array([0.1 * feasibility_tol, feasibility_tol, centrality_tol]),
        nswp=nwsp,
        aux_matrix_blocks=lag_maps,
        rmax=Rmax,
        local_solver=lambda prev_sol, lhs, rhs, local_auxs: ipm_solve_local_system(prev_sol, lhs, rhs, local_auxs, eps=eps, num_blocks=num_blocks),
        error_func=error_func,
        verbose=verbose,
        rank_weighted_error=True
    )
    error_func = lambda lhs, sol, rhs: ipm_error(lhs, sol, rhs, num_blocks=num_blocks)
    lhs_skeleton = {}
    lin_op_tt_adj = tt_transpose(lin_op_tt)
    lhs_skeleton[(0, 1)] = tt_scale(-1, lin_op_tt_adj)
    lhs_skeleton[(1, 0)] = tt_scale(-1, lin_op_tt)
    lin_op_tt_ineq_adj = None
    if active_ineq:
        lin_op_tt_ineq_adj = tt_scale(-1, tt_transpose(lin_op_tt_ineq))
        lhs_skeleton[(0, 2)] = tt_rank_reduce(tt_reshape(tt_scale(-1, lin_op_tt_ineq_adj), (4, 4)), eps=op_tol)
        lhs_skeleton[(0, 3)] = tt_reshape(tt_identity(2 * dim), (4, 4))
        bias_tt_ineq = tt_rank_reduce(tt_reshape(bias_tt_ineq, (4, )), eps=op_tol)
    else:
        lhs_skeleton[(0, 2)] = tt_reshape(tt_identity(2 * dim), (4, 4))
    X_tt = tt_identity(dim)
    Y_tt = tt_zero_matrix(2*dim)
    T_tt = None
    if active_ineq:
        T_tt = tt_reshape(tt_fast_matrix_vec_mul(lin_op_tt_ineq_adj, tt_reshape(tt_one_matrix(dim), (4, )), eps), (2, 2))
        T_tt = tt_normalise(T_tt)
    Z_tt = tt_identity(dim)
    iter = 0
    last = False
    prev_pd_error = np.inf
    for iter in range(1, max_iter):
        x_step_size, z_step_size, Delta_X_tt, Delta_Y_tt, Delta_Z_tt, pd_error, mu, sigma = _tt_ipm_newton_step(
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
            Y_tt,
            T_tt,
            Z_tt,
            op_tol,
            feasibility_tol,
            eps,
            active_ineq,
            solver,
            direction,
        )
        if (np.less(pd_error, feasibility_tol) or np.less(np.abs(prev_pd_error - pd_error), op_tol)) and np.less(mu, 1e-3):
            last = True
        #print("Before", scip.linalg.eigvalsh(tt_matrix_to_matrix(tt_add(X_tt, tt_scale(x_step_size, Delta_X_tt))))[0])
        X_tt = tt_rank_reduce(tt_add(X_tt, tt_scale(x_step_size, Delta_X_tt)), eps=op_tol, rank_weighted_error=True)
        #print("After", scip.linalg.eigvalsh(tt_matrix_to_matrix(X_tt))[0])
        Y_tt = tt_add(Y_tt, tt_scale(z_step_size, Delta_Y_tt))
        Y_tt = tt_rank_reduce(tt_sub(Y_tt, tt_reshape(tt_fast_matrix_vec_mul(lag_maps["y"], tt_reshape(Y_tt, shape=(4, )), eps), (2, 2))), eps=op_tol, rank_weighted_error=True)
        Z_tt = tt_rank_reduce(tt_add(Z_tt, tt_scale(z_step_size, Delta_Z_tt)), eps=op_tol, rank_weighted_error=True)

        if verbose:
            print(f"---Step {iter}---")
            print(f"Step sizes: {x_step_size}, {z_step_size}")
            print(f"Duality Gap: {100 * np.abs(mu):.4f}%")
            print(f"Primal-Dual error: {pd_error:.8f}")
            print(f"Sigma: {sigma:.4f}")
            print(
                f"Ranks X_tt: {tt_ranks(X_tt)}, Z_tt: {tt_ranks(Z_tt)}, \n"
                f"      Y_tt: {tt_ranks(Y_tt)}, T_tt: {tt_ranks(T_tt) if active_ineq else None} \n"
            )
        if last:
            break
        prev_pd_error = pd_error
    if verbose:
        print(f"---Terminated---")
        print(f"Converged in {iter} iterations.")
    return X_tt, Y_tt, T_tt, Z_tt