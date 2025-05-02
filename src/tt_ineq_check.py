import copy
import sys
import os
import time

import scipy.sparse.linalg

sys.path.append(os.getcwd() + '/../')

from src.tt_ops import *
from cy_src.ops_cy import *
from src.tt_amen import _compute_phi_bck_A, cached_einsum



def _tt_is_psd(A, nswp=10, x0=None, eps=1e-10, verbose=False):
    if verbose:
        print(f"Starting Eigen solve with:\n \t {eps} \n \t sweeps: {nswp}")
        t0 = time.time()
    dtype = A[0].dtype
    damp = 2

    min_or_max = "SA"

    if x0 == None:
        x_cores = [np.ones_like(c[:, :, 0], dtype=dtype) for c in A]
    else:
        x_cores = x0.copy()

    d = len(x_cores)
    rx = np.array([1] + tt_ranks(x_cores) + [1])
    N = np.array([c.shape[1] for c in x_cores])

    XAX = [np.ones((1, 1, 1), dtype=dtype)] + [None] * (d - 1) + [np.ones((1, 1, 1), dtype=dtype)]  # size is rk x Rk x rk
    real_tol = (eps / np.sqrt(d)) / damp
    max_res = 0
    for swp in range(nswp):
        x_cores = tt_rl_orthogonalise(x_cores)
        rx[1:-1] = np.array(tt_ranks(x_cores))
        XAX, no = compute_phi_bcks_A(XAX, x_cores, A, x_cores, d=d)

        # start loop
        max_res = 0

        for k in range(d):
            previous_solution = np.reshape(x_cores[k], (-1, 1))

            # solve the local system
            B = einsum("lsr,smnS,LSR->lmLrnR", XAX[k], A[k], XAX[k + 1], optimize=True)
            B = np.reshape(B, [rx[k] * N[k] * rx[k + 1], rx[k] * N[k] * rx[k + 1]])

            #TODO: Need to normalise x_cores ?
            eig_val, solution_now = scip.sparse.linalg.eigsh(B, k=1, which=min_or_max, v0=previous_solution)
            if np.less(eig_val, -2*eps):
                return False, 0

            norm_rhs = eig_val if abs(eig_val) > real_tol else 1.0
            res_new = np.linalg.norm(B @ solution_now - eig_val * solution_now) / norm_rhs
            res_old = np.linalg.norm(B @ previous_solution - eig_val * previous_solution) / norm_rhs

            max_res = max(res_old, res_new)

            solution_now = np.reshape(solution_now, (rx[k] * N[k], rx[k + 1]))
            # truncation
            if k < d - 1:
                u, s, v = scip.linalg.svd(solution_now, full_matrices=False, check_finite=False)
                r = 0
                for r in range(u.shape[1] - 1, 0, -1):
                    # solution has the same size
                    solution = np.reshape(u[:, :r] @ np.diag(s[:r]) @ v[:r, :], [-1, 1])
                    res = np.linalg.norm(B @ solution - eig_val * solution) / norm_rhs
                    if res > max(real_tol * damp, res_new):
                        break
                r += 1

                r = min(r, np.size(s))
            else:
                u, v = np.linalg.qr(solution_now)
                r = u.shape[1]
                s = np.ones(r, dtype=dtype)

            u = u[:, :r]
            v = np.diag(s[:r]) @ v[:r, :]
            v = v.T


            if k < d - 1:
                v = einsum('ji,jkl->ikl', v, x_cores[k + 1], optimize=True)

                x_cores[k] = np.reshape(u, [rx[k], N[k], r])
                x_cores[k + 1] = np.reshape(v, [r, N[k + 1], rx[k + 2]])
                rx[k + 1] = r

                # next phis with norm correction
                XAX[k + 1] = compute_phi_fwd_A(XAX[k], x_cores[k], A[k], x_cores[k])

                # ... and norms
                norm = np.linalg.norm(XAX[k + 1])
                norm = norm if np.greater(norm, 0) else 1.0
                XAX[k + 1] = np.divide(XAX[k + 1], norm)

            else:
                x_cores[k] = np.reshape(u @ v.T, (rx[k], N[k], rx[k + 1]))

        x_cores = tt_normalise(x_cores)

        if max_res < eps:
            break

    if verbose:
        print("\t -----")
        print(f"\t Solution rank is {rx[1:-1]}")
        print(f"\t Residual {max_res}")
        print('\t Number of sweeps', swp + 1)
        print('\t Time: ', time.time() - t0)
        print('\t Time per sweep: ', (time.time() - t0) / (swp + 1))

    final_eig_val = tt_inner_prod(x_cores, tt_fast_matrix_vec_mul(A, x_cores))
    return np.greater(final_eig_val, -0.5*eps), max_res


def tt_is_psd(A, op_tol, degenerate=False, eps=1e-10, verbose=False):
    if degenerate:
        A = tt_add(A, tt_scale(eps, tt_identity(len(A))))
    A = tt_rank_reduce(A, op_tol, rank_weighted_error=True)
    return _tt_is_psd(A, eps=eps, verbose=verbose)


def tt_is_geq(linear_op_tt, X_tt, vec_b_tt, op_tol, nswp=10, eps=1e-10, degenerate=False, verbose=False):
    res_tt = tt_sub(vec_b_tt, tt_fast_matrix_vec_mul(linear_op_tt, tt_split_bonds(X_tt)))
    norm = np.sqrt(tt_inner_prod(res_tt, res_tt))
    if norm > eps:
        res_tt = tt_scale(np.divide(2, norm), res_tt)
        A = tt_diag(res_tt)
        if degenerate:
            A = tt_add(A, tt_scale(2*eps, tt_identity(len(A))))
        A = tt_rank_reduce(A, op_tol, rank_weighted_error=True)
        return tt_pd_optimal_step_size(A, nswp=nswp, tol=eps, verbose=verbose)
    return True, 0.0


def tt_is_geq_zero(X_tt, op_tol, nswp=10, eps=1e-10, degenerate=False, verbose=False):
    res_tt = tt_split_bonds(X_tt)
    norm = np.sqrt(tt_inner_prod(res_tt, res_tt))
    if norm > eps:
        res_tt = tt_scale(np.divide(2, norm), res_tt)
        A = tt_diag(res_tt)
        if degenerate:
            A = tt_add(A, tt_scale(2*eps, tt_identity(len(A))))
        A = tt_rank_reduce(A, op_tol, rank_weighted_error=True)
        return tt_pd_optimal_step_size(A, nswp=nswp, tol=eps, verbose=verbose)
    return True, 0.0


def symmetric_powers_of_two(length):
    half = length // 2
    first_half = [2**i for i in range(1, half + 1)]
    if length % 2 == 0:
        return first_half + first_half[::-1]
    else:
        return first_half + [2**(half + 1)] + first_half[::-1]




def _step_size_local_solve(previous_solution, XDX_k, Delta_k, XDX_k1, XAX_k, A_k, XAX_k1, m, step_size, op_tol, size_limit, eps):
    if m <= size_limit:
        previous_solution = previous_solution.reshape(-1, 1)
        D = cached_einsum(
            "lsr,smnS,LSR->lmLrnR",
            XDX_k, Delta_k, XDX_k1
        ).reshape(m, m)
        A = cached_einsum("lsr,smnS,LSR->lmLrnR", XAX_k, A_k, XAX_k1).reshape(m, m)
        try:
            eig_val, solution_now = scip.sparse.linalg.eigsh((1/step_size)*A + D, tol=eps, k=1, which="SA", v0=previous_solution)
        except Exception as e:
            eig_val = previous_solution.T @ ((1/step_size)*A + D)  @ previous_solution
            solution_now = previous_solution
        if eig_val < 0:
            try:
                eig_val, solution_now = scip.sparse.linalg.eigsh(-D, M=A, tol=eps, k=1, which="LA", v0=previous_solution)
                step_size = max(0, min(step_size, (1 - op_tol) / eig_val[0]))
            except Exception as e:
                solution_now = previous_solution
        old_res = np.linalg.norm(previous_solution.reshape(-1, 1).T @ ((1/step_size)*A + D) @ previous_solution * previous_solution.reshape(-1, 1) - ((1/step_size)*A + D) @ previous_solution)
        return solution_now, step_size, old_res

    x_shape = previous_solution.shape
    mat_vec_A = lambda x_vec: cached_einsum('lsr,smnS,LSR,rnR->lmL',XAX_k, A_k, XAX_k1, x_vec.reshape(*x_shape)).reshape(-1, 1)
    A_op = scip.sparse.linalg.LinearOperator((m, m), matvec=mat_vec_A)
    mat_vec_D = lambda x_vec: -cached_einsum('lsr,smnS,LSR,rnR->lmL', XDX_k, Delta_k, XDX_k1, x_vec.reshape(*x_shape)).reshape(-1, 1)
    D_op = scip.sparse.linalg.LinearOperator((m, m), matvec=mat_vec_D)
    AD_op = scip.sparse.linalg.LinearOperator((m, m), matvec=lambda x_vec: mat_vec_A(x_vec) / step_size - mat_vec_D(x_vec))

    try:
        eig_val, solution_now = scip.sparse.linalg.eigsh(AD_op, tol=eps, k=1, which="SA", v0=previous_solution)
    except Exception as e:
        eig_val = previous_solution.reshape(-1, 1).T @ AD_op(previous_solution)
        solution_now = previous_solution
    if eig_val < 0:
        try:
            eig_val, solution_now = scip.sparse.linalg.eigsh(D_op, M=A_op, tol=eps, k=1, which="LA", v0=previous_solution)
            step_size = max(0, min(step_size, (1 - op_tol) / eig_val[0]))
        except Exception as e:
            solution_now = previous_solution

    old_res = np.linalg.norm(previous_solution.reshape(-1, 1).T @ AD_op(previous_solution) * previous_solution.reshape(-1, 1) - AD_op(previous_solution))

    return solution_now.reshape(-1, 1), step_size, old_res


def tt_ineq_optimal_step_size(A, Delta, op_tol, nswp=10, verbose=False):
    x_cores =  tt_random_gaussian([2]*(len(A)-1), (2,))
    return tt_pd_optimal_step_size(A, Delta, op_tol, kick_rank=max(int(2 ** (len(A) / 4) / (2 * nswp)), 1), x0=x_cores, nswp=nswp, tol=0.1*op_tol, verbose=verbose)


def _add_kick_rank(u, v, r_add=2):
    old_r = u.shape[-1]
    uk = np.random.randn(u.shape[0], r_add)  # rx_k x N_k x rz_k+1
    u, Rmat = scp.linalg.qr(np.concatenate((u, uk), 1), check_finite=False, mode="economic", overwrite_a=True)
    v = Rmat[:, :old_r] @ v
    return u, v, u.shape[-1]


def tt_pd_optimal_step_size(A, Delta, op_tol, x0=None, kick_rank=None, nswp=10, tol=1e-12, verbose=False):
    if verbose:
        print(f"Starting Eigen solve with:\n \t {tol} \n \t sweeps: {nswp}")
        t0 = time.time()
    if x0 is None:
        x_cores = tt_random_gaussian([2]*(len(A)-1), (2,))
    else:
        x_cores = x0
    if kick_rank is None:
        kick_rank = max(int(2 ** (len(A) / 2) / (2 * nswp)), 1)
    d = len(x_cores)
    rx = np.array([1] + tt_ranks(x_cores) + [1])
    N = np.array([c.shape[1] for c in x_cores])

    XAX = [np.ones((1, 1, 1))] + [None] * (d - 1) + [np.ones((1, 1, 1))]  # size is rk x Rk x rk
    XDX = copy.deepcopy(XAX)

    max_res = 0
    step_size = 1
    last = False
    size_limit = 0.4 * N[0] * (int(np.sqrt(d) * d))**2
    for swp in range(nswp):
        max_res = np.inf if swp == 0 else 0
        for k in range(d - 1, -1, -1):
            if swp > 0:
                previous_solution = x_cores[k]
                solution_now, step_size, local_res = _step_size_local_solve(previous_solution, XDX[k], Delta[k], XDX[k+1], XAX[k], A[k], XAX[k+1], rx[k] * N[k] * rx[k + 1], step_size, op_tol, size_limit, tol)
                max_res = max(max_res, local_res)
                solution_now = np.reshape(solution_now, (rx[k], N[k] * rx[k + 1])).T
            else:
                solution_now = np.reshape(x_cores[k], (rx[k], N[k] * rx[k + 1])).T

            if k > 0:
                u, s, v = scip.linalg.svd(solution_now, full_matrices=False, check_finite=False, overwrite_a=True)
                v = s.reshape(-1, 1) * v
                r = prune_singular_vals(s, 0.5*tol)
                if not last:
                    u, v, r = _add_kick_rank(u[:, :r], v[:r], kick_rank)
                else:
                    u = u[:, :r]
                    v = v[:r]
                x_cores[k] = np.reshape(u.T, (r, N[k], rx[k + 1]))
                x_cores[k - 1] = einsum('rdc,cR->rdR', x_cores[k - 1], v.T, optimize="greedy")
                rx[k] = r

                XAX[k] = _compute_phi_bck_A(XAX[k + 1], x_cores[k], A[k], x_cores[k])
                XDX[k] = _compute_phi_bck_A(XDX[k + 1], x_cores[k], Delta[k], x_cores[k])
                norm = np.sqrt(np.linalg.norm(XAX[k]) ** 2 + np.linalg.norm(XDX[k]) ** 2)
                norm = norm if norm > 0 else 1.0
                XAX[k] = np.divide(XAX[k], norm)
                XDX[k] = np.divide(XDX[k], norm)

            else:
                x_cores[k] = np.reshape(solution_now, (rx[k], N[k], rx[k + 1]))

        x_cores = tt_normalise(x_cores)
        if step_size == 0 or last:
            break
        if max_res < tol or swp == nswp - 1:
            last = True
        if verbose:
            print('Starting Sweep:\n\tMax num of sweeps: %d' % swp)
            print(f"\tDirection: {-1}")
            print(f'\tResidual {max_res}')
            print(f"\tTT-sol rank: {tt_ranks(x_cores)}", flush=True)
        max_res = 0
        for k in range(d):
            previous_solution = x_cores[k]
            solution_now, step_size, local_res = _step_size_local_solve(previous_solution, XDX[k], Delta[k], XDX[k+1], XAX[k], A[k], XAX[k+1], rx[k] * N[k] * rx[k + 1], step_size, op_tol, size_limit, tol)
            max_res = max(max_res, local_res)
            solution_now = np.reshape(solution_now, (rx[k] * N[k], rx[k + 1]))
            if k < d - 1:
                u, s, v = scip.linalg.svd(solution_now, full_matrices=False, check_finite=False, overwrite_a=True)
                v = s.reshape(-1, 1) * v
                r = prune_singular_vals(s, 0.5*tol)
                if not last:
                    u, v, r = _add_kick_rank(u[:, :r], v[:r, :], kick_rank)
                else:
                    u = u[:, :r]
                    v = v[:r, :]
                x_cores[k] = u.reshape(rx[k], N[k], r)
                x_cores[k + 1] = einsum('ij,jkl->ikl', v, x_cores[k + 1], optimize="greedy").reshape(r, N[k + 1], rx[k + 2])
                rx[k + 1] = r
                XAX[k + 1] = compute_phi_fwd_A(XAX[k], x_cores[k], A[k], x_cores[k])
                XDX[k + 1] = compute_phi_fwd_A(XDX[k], x_cores[k], Delta[k], x_cores[k])
                norm = np.sqrt(np.linalg.norm(XAX[k + 1]) ** 2 + np.linalg.norm(XDX[k + 1]) ** 2)
                norm = norm if np.greater(norm, 0) else 1.0
                XAX[k + 1] = np.divide(XAX[k + 1], norm)
                XDX[k + 1] = np.divide(XDX[k + 1], norm)
            else:
                x_cores[k] = np.reshape(solution_now, (rx[k], N[k], rx[k + 1]))

        x_cores = tt_normalise(x_cores)
        if step_size == 0 or last:
            break
        if max_res < tol:
            last = True
        if verbose:
            print('Starting Sweep:\n\tMax num of sweeps: %d' % swp)
            print(f"\tDirection: {1}")
            print(f'\tResidual {max_res}')
            print(f"\tTT-sol rank: {tt_ranks(x_cores)}", flush=True)

    if verbose:
        print("\t -----")
        print(f"\t Solution rank is {rx[1:-1]}")
        print(f"\t Residual {max_res}")
        print('\t Number of sweeps', swp + 1)
        print('\t Time: ', time.time() - t0)
        print('\t Time per sweep: ', (time.time() - t0) / (swp + 1))

    min_eig_value = tt_inner_prod(x_cores, tt_fast_matrix_vec_mul(A, x_cores, tol)) + step_size * tt_inner_prod(x_cores, tt_fast_matrix_vec_mul(Delta, x_cores, tol))
    return step_size, max(min_eig_value, 0)


def tt_psd_rank_reduce(A, eigen_tt, op_tol, rank_weighted_error=False):
    d = len(eigen_tt)
    rx = np.array([1] + tt_ranks(eigen_tt) + [1])
    N = np.array([c.shape[1] for c in eigen_tt])

    ranks = np.array([1] + tt_ranks(A) + [1])
    if d == 1 or np.all(ranks == 1):
        return A
    if rank_weighted_error:
        weights = ranks[1:]*ranks[:-1]
        op_tol = np.sqrt(weights/np.sum(weights))*op_tol
    else:
        op_tol = np.ones(d - 1) * (op_tol / np.sqrt(d - 1))

    XAX = [np.ones((1, 1, 1))] + [None] * (d - 1) + [np.ones((1, 1, 1))]  # size is rk x Rk x rk

    rank = 1
    singular_values = {}
    for idx, tt_core in enumerate(A[:-1]):
        idx_shape = tt_core.shape
        next_idx_shape = A[idx + 1].shape
        k = len(idx_shape) - 1
        u, s, v_t = scp.linalg.svd(A[idx].reshape(rank * np.prod(idx_shape[1:k], dtype=int), -1), full_matrices=False, check_finite=False)
        singular_values[idx] = s
        next_rank = u.shape[-1]
        A[idx] = u.reshape(rank, *idx_shape[1:-1], next_rank)
        A[idx + 1] = (
                s.reshape(-1, 1) * v_t @ A[idx + 1].reshape(v_t.shape[-1], -1)
        ).reshape(next_rank, *next_idx_shape[1:-1], -1)
        rank = next_rank

        x_cores = tt_rl_orthogonalise(x_cores)
        rx[1:-1] = np.array(tt_ranks(x_cores))
    for k in range(d - 1, 0, -1):
        XAX[k] = _compute_phi_bck_A(
            XAX[k + 1], x_cores[k], A[k], x_cores[k])
        norm = np.linalg.norm(XAX[k])
        norm = norm if norm > 0 else 1.0
        XAX[k + 1] = np.divide(XAX[k + 1], norm)
    next_indices = np.array([0])
    for k in range(d):
        XAX[k] = XAX[k][:, None, next_indices, :]
        A[k] = A[k][None, next_indices]
        A_bar = einsum(
            "lsr,smnS,LSR->SlmLrnR",
            XAX[k], A[k], XAX[k + 1],
            optimize=True
        ).reshape(A[k].shape[-1], rx[k] * N[k] * rx[k + 1], rx[k] * N[k] * rx[k + 1])

        previous_solution = x_cores[k].flattten()

        local_eigs = einsum("m, Smn, n -> S", previous_solution, A_bar, previous_solution)
        s = singular_values[k]
        neg_indices = np.argwhere(local_eigs < 0)
        sc = np.cumsum(np.abs(s[neg_indices][::-1]) ** 2)[::-1]
        next_indices_neg = neg_indices[:np.argmax(sc < op_tol[k] ** 2)]
        temp_tol_sq = np.sum(np.abs(s[next_indices_neg]**2))

        pos_indices = np.argwhere(1-(local_eigs < 0))
        sc = np.cumsum(np.abs(s[pos_indices][::-1]) ** 2)[::-1]
        next_indices_pos = pos_indices[:np.argmax(sc < min(temp_tol_sq, op_tol[k]**2 - temp_tol_sq))]
        next_indices = np.sort(np.concatenate((next_indices_neg, next_indices_pos)))
        A[k] = A[k].reshape(rank * np.prod(idx_shape[1:k], dtype=int), -1)[:, :, :, None, next_indices]

    return A
