import time

import numpy as np
import scipy as scp
from src.tt_ops import *
from src.tt_ops import _tt_core_collapse, _block_diag_tensor, _tt_mat_mat_collapse
from tqdm import tqdm
import copy


def _tt_burer_monteiro_grad(K_orig_qp, A_44, S_1, S_2, S_3, V_00, V_01, V_10, V_11, orig_p, orig_q, p, q):
    pair_1 = np.kron(V_00, V_00) + np.kron(V_10, V_10)
    pair_2 = np.kron(V_01, V_00) + np.kron(V_11, V_10)
    pair_3 = np.kron(V_00, V_01) + np.kron(V_10, V_11)

    D_1 = (
              np.trace((pair_1 @ A_44.reshape(q, q * p ** 2)).reshape(p, p, p, q, order="F"), axis1=0, axis2=2)
              + np.trace((A_44.reshape(p * q ** 2, p) @ pair_1).reshape(p, q, q, q, order="F"), axis1=1, axis2=3)
          ).reshape(p * q, 1, order="F") + S_1
    D_2 = (
              np.trace((pair_2 @ A_44.reshape(q, q * p ** 2)).reshape(p, p, p, q, order="F"), axis1=0, axis2=2)
              + np.trace((A_44.reshape(p * q ** 2, p) @ pair_2).reshape(p, q, q, q, order="F"), axis1=1, axis2=3)
          ).reshape(p * q, 1, order="F") + S_2
    D_3 = (
              np.trace((pair_3 @ A_44.reshape(q, q * p ** 2)).reshape(p, p, p, q, order="F"), axis1=0, axis2=2)
              + np.trace((A_44.reshape(p * q ** 2, p) @ pair_3).reshape(p, q, q, q, order="F"), axis1=1, axis2=3)
          ).reshape(p * q, 1, order="F") + S_3

    H_0 = D_1.reshape(orig_p, q * orig_p, order="F") @ np.einsum('ijt, jl->ilt',
                                                                 K_orig_qp.reshape(orig_q * orig_p, orig_p, orig_q),
                                                                 V_00).reshape(orig_p * q, orig_q, order="F")
    H_1 = D_2.reshape(orig_p, q * orig_p, order="F") @ np.einsum('ijt, jl->ilt',
                                                                 K_orig_qp.reshape(orig_q * orig_p, orig_p, orig_q),
                                                                 V_01).reshape(orig_p * q, orig_q, order="F")
    G_0 = np.einsum('ijt, lj->ilt', K_orig_qp.reshape(orig_q * orig_p, orig_q, orig_p, order="F"), V_00).reshape(
        orig_q * p, orig_p).T @ D_1.reshape(orig_q * p, orig_q, order="F")
    G_1 = np.einsum('ijt, lj->ilt', K_orig_qp.reshape(orig_q * orig_p, orig_q, orig_p, order="F"), V_01).reshape(
        orig_q * p, orig_p).T @ D_3.reshape(orig_q * p, orig_q, order="F")

    return G_0 + H_0 + H_1 + G_1


def _tt_bm_backtracking(
    A_22, A_33, A_44,
    C_00, C_01, C_10, C_11,
    V_00, V_01, V_10, V_11,
    v_00_grad, v_01_grad, v_10_grad, v_11_grad,
    vec_00, vec_01, vec_10, vec_11,
    lr, beta, gamma,
    num_swps=5,
    c=1e-4
):
    armijo_term = -c*(
        np.trace(vec_00.T @ vec_00)
        + np.trace(vec_01.T @ vec_01)
        + np.trace(vec_10.T @ vec_10)
        + np.trace(vec_11.T @ vec_11)
    )
    pair_1 = np.kron(V_00, V_00) + np.kron(V_10, V_10)
    pair_2 = np.kron(V_01, V_00) + np.kron(V_11, V_10)
    pair_3 = np.kron(V_00, V_01) + np.kron(V_10, V_11)
    pair_4 = np.kron(V_01, V_01) + np.kron(V_11, V_11)
    obj_val = (
        np.trace(A_22 @ (np.kron(C_00, pair_1) + np.kron(C_01, pair_2) + np.kron(C_10, pair_3) + np.kron(C_11, pair_4)))
        + np.trace(
        A_33 @ (np.kron(pair_1, C_00) + np.kron(pair_2, C_01) + np.kron(pair_3, C_10) + np.kron(pair_4, C_11)))
        + np.trace(
        A_44 @ (np.kron(pair_1, pair_1) + np.kron(pair_2, pair_2) + np.kron(pair_3, pair_3) + np.kron(pair_4, pair_4)))
    )
    for _ in range(num_swps):
        v_00_grad_l = gamma * v_00_grad + lr * vec_00
        v_01_grad_l = gamma * v_01_grad + lr * vec_01
        v_10_grad_l = gamma * v_10_grad + lr * vec_10
        v_11_grad_l = gamma * v_11_grad + lr * vec_11
        pair_1 = np.kron(V_00 - v_00_grad_l, V_00 - v_00_grad_l) + np.kron(V_10 - v_10_grad_l, V_10 - v_10_grad_l)
        pair_2 = np.kron(V_01 - v_01_grad_l, V_00 - v_00_grad_l) + np.kron(V_11 - v_11_grad_l, V_10 - v_10_grad_l)
        pair_3 = np.kron(V_00 - v_00_grad_l, V_01 - v_01_grad_l) + np.kron(V_10 - v_10_grad_l, V_11 - v_11_grad_l)
        pair_4 = np.kron(V_01 - v_01_grad_l, V_01 - v_01_grad_l) + np.kron(V_11 - v_11_grad_l, V_11 - v_11_grad_l)
        new_obj_val = (
            np.trace(A_22 @ (np.kron(C_00, pair_1) + np.kron(C_01, pair_2) + np.kron(C_10, pair_3) + np.kron(C_11, pair_4)))
            + np.trace(A_33 @ (np.kron(pair_1, C_00) + np.kron(pair_2, C_01) + np.kron(pair_3, C_10) + np.kron(pair_4, C_11)))
            + np.trace(A_44 @ (np.kron(pair_1, pair_1) + np.kron(pair_2, pair_2) + np.kron(pair_3, pair_3) + np.kron(pair_4, pair_4)))
        )
        if np.less(new_obj_val, obj_val + lr*armijo_term):
            break
        else:
            lr *= beta

    return v_00_grad_l, v_01_grad_l, v_10_grad_l, v_11_grad_l, lr


def _tt_bm_core_wise(matrix_tt, factor_tt, A_22, A_33, A_44, idx, lr, is_block=False, beta=0.9, num_swps=20, gamma=0.9,
                     tol=1e-5):
    """ Local Gradient Descent iteration with Nesterov acceleration """
    xr_i, _, _, xr_ip1 = factor_tt[idx].shape

    C_00 = matrix_tt[idx][:, 0, 0]
    C_01 = matrix_tt[idx][:, 0, 1]
    C_10 = matrix_tt[idx][:, 1, 0]
    C_11 = matrix_tt[idx][:, 1, 1]
    V_00 = factor_tt[idx][:, 0, 0]
    V_01 = factor_tt[idx][:, 0, 1]
    V_10 = factor_tt[idx][:, 1, 0]
    V_11 = factor_tt[idx][:, 1, 1]
    if idx == 0 or idx == len(factor_tt) - 1:
        xr_i = max(xr_i, xr_ip1)
        xr_ip1 = max(xr_i, xr_ip1)
        C_00 = np.diag(C_00.flatten())
        C_10 = np.diag(C_10.flatten())
        C_01 = np.diag(C_01.flatten())
        C_11 = np.diag(C_11.flatten())
        V_00 = np.diag(V_00.flatten())
        V_10 = np.diag(V_10.flatten())
        V_01 = np.diag(V_01.flatten())
        V_11 = np.diag(V_11.flatten())

    m, n = C_00.shape
    orig_p, orig_q = V_00.shape
    p = orig_p ** 2
    q = orig_q ** 2
    max_s = max(q * m, n * p)
    I = np.eye(max_s)
    indices = np.arange(max_s)
    K_np = I[indices[:n * p].reshape((n, p), order="F").ravel(), :n * p]
    K_qm = I[indices[:q * m].reshape((q, m), order="F").ravel(), :q * m]
    K_orig_qp = I[indices[:orig_q * orig_p].reshape((orig_q, orig_p), order="F").ravel(), :orig_q * orig_p]
    S = (
        np.einsum('ijt, jl->ilt', A_22.T.reshape(p, m * q, n, order="F"), K_qm).reshape(p * q, m * n, order="F")
        + np.einsum('ijt, jl->ilt', A_33.T.reshape(m, n * p, q, order="F"), K_np).reshape(m * n, p * q, order="F").T
    )
    S_1 = S @ C_00.reshape(m * n, 1, order="F")
    S_2 = S @ C_01.reshape(m * n, 1, order="F")
    S_3 = S @ C_10.reshape(m * n, 1, order="F")
    S_4 = S @ C_11.reshape(m * n, 1, order="F")

    v_00_grad = 0
    v_01_grad = 0
    v_10_grad = 0
    v_11_grad = 0

    vec_01 = np.zeros((xr_ip1, xr_ip1), dtype=float)
    vec_10 = np.zeros((xr_ip1, xr_ip1), dtype=float)

    for swp in range(num_swps):
        V_00_nest = V_00 - gamma * v_00_grad
        V_01_nest = V_01 - gamma * v_01_grad
        V_10_nest = V_10 - gamma * v_10_grad
        V_11_nest = V_11 - gamma * v_11_grad

        vec_00 = _tt_burer_monteiro_grad(K_orig_qp, A_44, S_1, S_2, S_3, V_00_nest, V_01_nest, V_10_nest,
                                         V_11_nest, orig_p, orig_q, p, q)
        vec_11 = _tt_burer_monteiro_grad(K_orig_qp, A_44, S_4, S_3, S_2, V_11_nest, V_10_nest, V_01_nest,
                                         V_00_nest, orig_p, orig_q, p, q)

        max_norm = max(np.linalg.norm(vec_00), np.linalg.norm(vec_11))
        if idx != 0 or not is_block:
            vec_01 = _tt_burer_monteiro_grad(K_orig_qp, A_44, S_4, S_3, S_2, V_01_nest, V_00_nest, V_11_nest,
                                             V_10_nest, orig_p, orig_q, p, q)
            vec_10 = _tt_burer_monteiro_grad(K_orig_qp, A_44, S_1, S_2, S_3, V_10_nest, V_11_nest, V_00_nest,
                                             V_01_nest, orig_p, orig_q, p, q)
            max_norm = max(max_norm, np.linalg.norm(vec_01), np.linalg.norm(vec_10))

        # gradient clipping for numerical stability
        vec_00 = np.divide(vec_00, max_norm)
        vec_01 = np.divide(vec_01, max_norm)
        vec_10 = np.divide(vec_10, max_norm)
        vec_11 = np.divide(vec_11, max_norm)

        v_00_grad, v_01_grad, v_10_grad, v_11_grad, lr = _tt_bm_backtracking(
            A_22, A_33, A_44,
            C_00, C_01, C_10, C_11,
            V_00, V_01, V_10, V_11,
            v_00_grad, v_01_grad, v_10_grad, v_11_grad,
            vec_00, vec_01, vec_10, vec_11,
            lr, beta, gamma
        )
        V_00 -= v_00_grad
        V_01 -= v_01_grad
        V_10 -= v_10_grad
        V_11 -= v_11_grad

        local_err = np.divide(np.linalg.norm(v_00_grad) + np.linalg.norm(v_01_grad) + np.linalg.norm(v_10_grad) + np.linalg.norm(
            v_11_grad), 4)
        if np.less_equal(local_err, tol):
            break

    if 0 < idx < len(factor_tt) - 1:
        factor_tt[idx][:, 0, 0, :] = V_00
        factor_tt[idx][:, 0, 1, :] = V_01
        factor_tt[idx][:, 1, 0, :] = V_10
        factor_tt[idx][:, 1, 1, :] = V_11
    else:
        ax = 0 if idx == 0 else -1
        factor_tt[idx][:, 0, 0, :] = np.expand_dims(np.diagonal(V_00), axis=ax)
        factor_tt[idx][:, 0, 1, :] = np.expand_dims(np.diagonal(V_01), axis=ax)
        factor_tt[idx][:, 1, 0, :] = np.expand_dims(np.diagonal(V_10), axis=ax)
        factor_tt[idx][:, 1, 1, :] = np.expand_dims(np.diagonal(V_11), axis=ax)

    return factor_tt, lr


def _adjust_diff(matrix_tt, factor_tt, idx):
    comp_tt_idx = _tt_mat_mat_collapse(factor_tt[idx], np.swapaxes(factor_tt[idx], axis1=1, axis2=2))
    return _tt_core_collapse(matrix_tt[idx], comp_tt_idx), _tt_core_collapse(comp_tt_idx, matrix_tt[idx]), _tt_core_collapse(comp_tt_idx, comp_tt_idx)


def tt_burer_monteiro_factorisation(psd_tt, solution_tt=None, is_block=False, num_swps=20, max_iter=25, beta=0.9,
                                    tol=1e-5, verbose=False):
    train_tt = tt_scale(-1, psd_tt)
    target_ranks = [int(np.ceil(np.sqrt(r))) + 1 for r in tt_ranks(train_tt)]
    if solution_tt is None:
        solution_tt = tt_random_gaussian(target_ranks, shape=(2, 2))
    else:
        add_on_ranks = [max(r - c_r, 1) for c_r, r in zip(tt_ranks(solution_tt), target_ranks)]
        solution_tt = tt_rank_retraction(solution_tt, [r - ar for ar, r in zip(add_on_ranks, target_ranks)])
        solution_tt = tt_add(solution_tt, tt_random_gaussian(add_on_ranks, shape=(2, 2)))
    solution_tt = tt_rl_orthogonalise(solution_tt)
    if is_block:
        solution_tt[0][:, 0, 1] = 0
        solution_tt[0][:, 1, 0] = 0
    lr = np.array([0.1] * len(train_tt))
    comp_tt = tt_mat_mat_mul(solution_tt, tt_transpose(solution_tt))
    A_44s = [_tt_core_collapse(c, c) for c in comp_tt]
    A_33s = [_tt_core_collapse(a, c) for a, c in zip(train_tt, comp_tt)]
    A_22s = [_tt_core_collapse(c, a) for c, a in zip(comp_tt, train_tt)]
    A_11_constant = tt_inner_prod(train_tt, train_tt)

    terminal_idx = len(solution_tt) - 1
    err = A_11_constant + safe_multi_dot(A_22s) + safe_multi_dot(A_33s) + safe_multi_dot(A_44s)
    if verbose:
        print(f"Initial Error: {err}, {lr}")
    for iteration in range(max_iter):
        left_contraction_A_22 = 1
        left_contraction_A_33 = 1
        left_contraction_A_44 = 1
        for k in range(terminal_idx):
            solution_tt = core_forward_orthogonalise(k, solution_tt)
            A_22s[k], A_33s[k], A_44s[k] = _adjust_diff(train_tt, solution_tt, k)
            left_contraction_A_22 = np.dot(left_contraction_A_22, A_22s[k])
            left_contraction_A_33 = np.dot(left_contraction_A_33, A_33s[k])
            left_contraction_A_44 = np.dot(left_contraction_A_44, A_44s[k])
            if k + 1 != terminal_idx:
                right_contraction_A_22 = safe_multi_dot(A_22s[k + 2:])
                right_contraction_A_33 = safe_multi_dot(A_33s[k + 2:])
                right_contraction_A_44 = safe_multi_dot(A_44s[k + 2:])

                A_22 = np.outer(right_contraction_A_22, left_contraction_A_22)
                A_33 = np.outer(right_contraction_A_33, left_contraction_A_33)
                A_44 = np.outer(right_contraction_A_44, left_contraction_A_44)
            else:
                A_22 = np.diag(left_contraction_A_22.flatten())
                A_33 = np.diag(left_contraction_A_33.flatten())
                A_44 = np.diag(left_contraction_A_44.flatten())
            solution_tt, lr_local = _tt_bm_core_wise(train_tt, solution_tt, A_22, A_33, A_44, k + 1, lr[k],
                                                  is_block=is_block,
                                                  beta=beta,
                                                  num_swps=num_swps, tol=0.1 * tol)
            lr[k] = beta * lr[k] + (1 - beta) * lr_local

        right_contraction_A_22 = 1
        right_contraction_A_33 = 1
        right_contraction_A_44 = 1
        for k in range(terminal_idx, 0, -1):
            solution_tt = core_backward_orthogonalise(k, solution_tt)
            A_22s[k], A_33s[k], A_44s[k] = _adjust_diff(train_tt, solution_tt, k)
            right_contraction_A_22 = np.dot(A_22s[k], right_contraction_A_22)
            right_contraction_A_33 = np.dot(A_33s[k], right_contraction_A_33)
            right_contraction_A_44 = np.dot(A_44s[k], right_contraction_A_44)
            if k - 1 != 0:
                left_contraction_A_22 = safe_multi_dot(A_22s[:k - 1])
                left_contraction_A_33 = safe_multi_dot(A_33s[:k - 1])
                left_contraction_A_44 = safe_multi_dot(A_44s[:k - 1])

                A_22 = np.outer(right_contraction_A_22, left_contraction_A_22)
                A_33 = np.outer(right_contraction_A_33, left_contraction_A_33)
                A_44 = np.outer(right_contraction_A_44, left_contraction_A_44)
            else:
                A_22 = np.diag(right_contraction_A_22.flatten())
                A_33 = np.diag(right_contraction_A_33.flatten())
                A_44 = np.diag(right_contraction_A_44.flatten())
            solution_tt, lr_local = _tt_bm_core_wise(train_tt, solution_tt, A_22, A_33, A_44, k - 1, lr[k],
                                                  is_block=is_block,
                                                  beta=beta,
                                                  num_swps=num_swps, tol=0.1 * tol)
            lr[k] = beta*lr[k] + (1-beta)*lr_local

        A_22s[0], A_33s[0], A_44s[0] = _adjust_diff(train_tt, solution_tt, 0)
        err = A_11_constant + A_22s[0] @ right_contraction_A_22 + A_33s[0] @ right_contraction_A_33 + A_44s[0] @ right_contraction_A_44
        if verbose:
            print(f"Error: {err}, {lr}")
        if np.less_equal(err, tol) or np.less_equal(np.max(lr), tol):
            break

    return solution_tt, err
