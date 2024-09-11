import numpy as np
cimport numpy as np
from src.ops import safe_multi_dot
from src.tt_ops import _tt_core_collapse, _block_diag_tensor, _tt_mat_mat_collapse, tt_scale, tt_mat_mat_mul, tt_ranks, tt_random_gaussian, tt_add, tt_transpose, tt_rl_orthogonalise, core_backward_orthogonalise, safe_multi_dot, core_forward_orthogonalise


def _tt_burer_monteiro_grad(
    np.ndarray[float, ndim=2] A_22, 
    np.ndarray[float, ndim=2] A_33, 
    np.ndarray[float, ndim=2] A_44, 
    np.ndarray[float, ndim=2] C_00, 
    np.ndarray[float, ndim=2] C_01, 
    np.ndarray[float, ndim=2] C_10, 
    np.ndarray[float, ndim=2] C_11, 
    np.ndarray[float, ndim=2] V_00, 
    np.ndarray[float, ndim=2] V_01, 
    np.ndarray[float, ndim=2] V_10, 
    np.ndarray[float, ndim=2] V_11
    ):
    cdef int m = C_00.shape[0]
    cdef int n = C_00.shape[1]
    cdef int orig_p = V_00.shape[0]
    cdef int orig_q = V_00.shape[1]
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
    pair_1 = np.kron(V_00, V_00) + np.kron(V_10, V_10)
    pair_2 = np.kron(V_01, V_00) + np.kron(V_11, V_10)
    pair_3 = np.kron(V_00, V_01) + np.kron(V_10, V_11)

    D_1 = (
              np.trace((pair_1 @ A_44.reshape(q, q * p ** 2)).reshape(p, p, p, q, order="F"), axis1=0, axis2=2)
              + np.trace((A_44.reshape(p * q ** 2, p) @ pair_1).reshape(p, q, q, q, order="F"), axis1=1, axis2=3)
          ).reshape(p * q, 1, order="F") + S @ C_00.reshape(m * n, 1, order="F")
    D_2 = (
              np.trace((pair_2 @ A_44.reshape(q, q * p ** 2)).reshape(p, p, p, q, order="F"), axis1=0, axis2=2)
              + np.trace((A_44.reshape(p * q ** 2, p) @ pair_2).reshape(p, q, q, q, order="F"), axis1=1, axis2=3)
          ).reshape(p * q, 1, order="F") + S @ C_01.reshape(m * n, 1, order="F")
    D_3 = (
              np.trace((pair_3 @ A_44.reshape(q, q * p ** 2)).reshape(p, p, p, q, order="F"), axis1=0, axis2=2)
              + np.trace((A_44.reshape(p * q ** 2, p) @ pair_3).reshape(p, q, q, q, order="F"), axis1=1, axis2=3)
          ).reshape(p * q, 1, order="F") + S @ C_10.reshape(m * n, 1, order="F")

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


def _tt_bm_core_wise(matrix_tt, factor_tt, A_22, A_33, A_44, idx, is_block=False, lr=0.5, num_swps=20, gamma=0.9,
                     tol=1e-5):
    """ Local Gradient Descent iteration with Nesterov acceleration """
    xr_i, _, _, xr_ip1 = factor_tt[idx].shape
    local_lr = lr

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

    v_00_grad = 0
    v_01_grad = 0
    v_10_grad = 0
    v_11_grad = 0

    vec_01 = np.zeros(xr_ip1 ** 2)
    vec_10 = np.zeros(xr_ip1 ** 2)

    for swp in range(num_swps):
        V_00_nest = V_00 - gamma * v_00_grad
        V_01_nest = V_01 - gamma * v_01_grad
        V_10_nest = V_10 - gamma * v_10_grad
        V_11_nest = V_11 - gamma * v_11_grad
        vec_00 = _tt_burer_monteiro_grad(A_22, A_33, A_44, C_00, C_01, C_10, C_11, V_00_nest, V_01_nest, V_10_nest,
                                         V_11_nest)
        vec_11 = _tt_burer_monteiro_grad(A_22, A_33, A_44, C_11, C_10, C_01, C_00, V_11_nest, V_10_nest, V_01_nest,
                                         V_00_nest)

        if idx != 0 or not is_block:
            vec_01 = _tt_burer_monteiro_grad(A_22, A_33, A_44, C_11, C_10, C_01, C_00, V_01_nest, V_00_nest, V_11_nest,
                                             V_10_nest)
            vec_10 = _tt_burer_monteiro_grad(A_22, A_33, A_44, C_00, C_01, C_10, C_11, V_10_nest, V_11_nest, V_00_nest,
                                             V_01_nest)

        v_00_grad = gamma * v_00_grad + local_lr * vec_00.reshape(xr_i, xr_ip1)
        v_01_grad = gamma * v_01_grad + local_lr * vec_01.reshape(xr_i, xr_ip1)
        v_10_grad = gamma * v_10_grad + local_lr * vec_10.reshape(xr_i, xr_ip1)
        v_11_grad = gamma * v_11_grad + local_lr * vec_11.reshape(xr_i, xr_ip1)
        V_00 -= v_00_grad
        V_01 -= v_01_grad
        V_10 -= v_10_grad
        V_11 -= v_11_grad

        local_err = (np.linalg.norm(v_00_grad) + np.linalg.norm(v_01_grad) + np.linalg.norm(v_10_grad) + np.linalg.norm(
            v_11_grad)) / 4
        local_lr *= 0.99
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
    if 0 < idx < len(factor_tt) - 1:
        diff_core_idx = _block_diag_tensor(matrix_tt[idx], comp_tt_idx)
    elif idx == 0:
        diff_core_idx = np.concatenate((matrix_tt[idx], comp_tt_idx), axis=-1)
    else:
        diff_core_idx = np.concatenate((matrix_tt[idx], comp_tt_idx), axis=0)
    return _tt_core_collapse(diff_core_idx, diff_core_idx)


def tt_burer_monteiro_factorisation(psd_tt, solution_tt=None, is_block=False, num_swps=20, max_iter=20, tol=1e-5):
    print("Using Cython!")
    train_tt = tt_scale(-1, psd_tt)
    target_ranks = [int(np.ceil(np.sqrt(r))) + 1 for r in tt_ranks(train_tt)]
    if solution_tt is None:
        solution_tt = tt_random_gaussian(target_ranks, shape=(2, 2))
    else:
        add_on_ranks = [max(r - c_r, 0) for c_r, r in zip(tt_ranks(solution_tt), target_ranks)]
        solution_tt = tt_add(solution_tt, tt_random_gaussian(add_on_ranks, shape=(2, 2)))
    solution_tt = tt_rl_orthogonalise(solution_tt)
    if is_block:
        solution_tt[0][:, 0, 1] = 0
        solution_tt[0][:, 1, 0] = 0
    lr = 0.2
    prev_err = 100
    comp_tt = tt_mat_mat_mul(solution_tt, tt_transpose(solution_tt))
    diff = [_tt_core_collapse(c, c) for c in tt_add(train_tt, comp_tt)]
    train_shapes = [(train_tt[0].shape[-1], train_tt[0].shape[-1])] + [(c.shape[0], c.shape[-1]) for c in
                                                                       train_tt[1:-1]] + [
                       (train_tt[-1].shape[0], train_tt[-1].shape[0])]
    solution_shapes = [(solution_tt[0].shape[-1] ** 2, solution_tt[0].shape[-1] ** 2)] + [
        (c.shape[0] ** 2, c.shape[-1] ** 2) for c in solution_tt[1:-1]] + [
                          (solution_tt[-1].shape[0] ** 2, solution_tt[-1].shape[0] ** 2)]
    index_set = [
        (
            (
                np.array([[i + (n + q) * j for i in range(q)] for j in range(n)]).flatten() + n,
                np.array([[i + (m + p) * j for i in range(p)] for j in range(m)]).flatten() + m
            ),
            (
                np.array([[i + (n + q) * j for i in range(n)] for j in range(q)]).flatten() + n * (n + q),
                np.array([[i + (m + p) * j for i in range(m)] for j in range(p)]).flatten() + m * (m + p)
            ),
            (
                np.array([[i + (n + q) * j for i in range(q)] for j in range(q)]).flatten() + n * (n + q + 1),
                np.array([[i + (m + p) * j for i in range(p)] for j in range(p)]).flatten() + m * (m + p + 1)
            )
        )
        for (m, n), (p, q) in zip(train_shapes, solution_shapes)
    ]
    terminal_idx = len(solution_tt) - 1
    for iteration in range(max_iter):
        left_contraction = 1
        for k in range(len(solution_tt) - 1):
            solution_tt = core_forward_orthogonalise(k, solution_tt)
            diff[k] = _adjust_diff(train_tt, solution_tt, k)
            left_contraction = np.dot(left_contraction, diff[k])
            if k + 1 != terminal_idx:
                right_contraction = safe_multi_dot(diff[k + 2:])
                A_22 = np.reshape(right_contraction, (-1, 1))[index_set[k + 1][0][0]] @ left_contraction.reshape(1, -1)[:,
                                                                                  index_set[k + 1][0][1]]
                A_33 = np.reshape(right_contraction, (-1, 1))[index_set[k + 1][1][0]] @ left_contraction.reshape(1, -1)[:,
                                                                                  index_set[k + 1][1][1]]
                A_44 = np.reshape(right_contraction, (-1, 1))[index_set[k + 1][2][0]] @ left_contraction.reshape(1, -1)[:,
                                                                                  index_set[k + 1][2][1]]
            else:
                A_22 = np.diag(left_contraction.flatten()[index_set[k + 1][0][1]])
                A_33 = np.diag(left_contraction.flatten()[index_set[k + 1][1][1]])
                A_44 = np.diag(left_contraction.flatten()[index_set[k + 1][2][1]])
            solution_tt, lr = _tt_bm_core_wise(train_tt, solution_tt, A_22, A_33, A_44, k + 1, is_block=is_block, lr=lr,
                                               num_swps=num_swps, tol=0.1 * tol)
        right_contraction = 1
        for k in range(terminal_idx, 0, -1):
            solution_tt = core_backward_orthogonalise(k, solution_tt)
            diff[k] = _adjust_diff(train_tt, solution_tt, k)
            right_contraction = np.dot(diff[k], right_contraction)
            if k - 1 != 0:
                left_contraction = np.reshape(safe_multi_dot(diff[:k - 1]), (1, -1))
                A_22 = right_contraction.reshape(-1, 1)[index_set[k - 1][0][0]] @ left_contraction.reshape(1, -1)[:,
                                                                                  index_set[k - 1][0][1]]
                A_33 = right_contraction.reshape(-1, 1)[index_set[k - 1][1][0]] @ left_contraction.reshape(1, -1)[:,
                                                                                  index_set[k - 1][1][1]]
                A_44 = right_contraction.reshape(-1, 1)[index_set[k - 1][2][0]] @ left_contraction.reshape(1, -1)[:,
                                                                                  index_set[k - 1][2][1]]
            else:
                A_22 = np.diag(right_contraction.flatten()[index_set[k - 1][0][1]])
                A_33 = np.diag(right_contraction.flatten()[index_set[k - 1][1][1]])
                A_44 = np.diag(right_contraction.flatten()[index_set[k - 1][2][1]])
            solution_tt, lr = _tt_bm_core_wise(train_tt, solution_tt, A_22, A_33, A_44, k - 1, is_block=is_block, lr=lr,
                                               num_swps=num_swps, tol=0.1 * tol)

        diff[0] = _adjust_diff(train_tt, solution_tt, 0)
        err = (diff[0] @ right_contraction).item()
        lr = min(0.99 * (prev_err / err) * lr, 0.2)
        prev_err = err
        print(f"Error: {err}, {lr}")
        if np.less_equal(err, tol):
            print(f"Converged in {iteration} iterations")
            break

    return solution_tt, prev_err