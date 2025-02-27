import sys
import os
import time

import numpy as np

sys.path.append(os.getcwd() + '/../')

from src.tt_ops import *
from src.tt_eig import tt_min_eig, tt_max_eig
from cy_src.ops_cy import *



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

    max_res = 0
    for swp in range(nswp):
        x_cores = tt_rl_orthogonalise(x_cores)
        rx[1:-1] = np.array(tt_ranks(x_cores))
        XAX, no = compute_phi_bcks_A(XAX, x_cores, A, x_cores, d=d)

        # start loop
        max_res = 0

        for k in range(d):
            previous_solution = np.reshape(x_cores[k], (-1, 1))
            # residuals
            real_tol = (eps / np.sqrt(d)) / damp

            # solve the local system
            Bp = einsum("smnS,LSR->smnRL", A[k], XAX[k + 1], optimize=True)
            B = einsum("lsr,smnRL->lmLrnR", XAX[k], Bp, optimize=True)
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

    final_eig_val = tt_inner_prod(x_cores, tt_matrix_vec_mul(A, x_cores))
    return np.greater(final_eig_val, -0.5*eps), max_res


def tt_is_psd(A, op_tol, degenerate=False, eps=1e-10, verbose=False):
    if degenerate:
        A = tt_add(A, tt_scale(eps, tt_identity(len(A))))
    A = tt_rank_reduce(A, op_tol, variable_error=True)
    return _tt_is_psd(A, eps=eps, verbose=verbose)


def tt_is_geq(linear_op_tt, X_tt, vec_b_tt, op_tol, nswp=10, eps=1e-10, degenerate=False, verbose=False):
    res_tt = tt_sub(vec_b_tt, tt_matrix_vec_mul(linear_op_tt, tt_vec(X_tt)))
    norm = np.sqrt(tt_inner_prod(res_tt, res_tt))
    if norm > eps:
        res_tt = tt_scale(np.divide(2, norm), res_tt)
        A = tt_diag(res_tt)
        if degenerate:
            A = tt_add(A, tt_scale(2*eps, tt_identity(len(A))))
        A = tt_rank_reduce(A, op_tol, variable_error=True)
        return _tt_is_psd(A, nswp=nswp, eps=eps, verbose=verbose)
    return True, 0.0


def tt_is_geq_(X_tt, op_tol, nswp=10, eps=1e-10, degenerate=False, verbose=False):
    res_tt = tt_vec(X_tt)
    norm = np.sqrt(tt_inner_prod(res_tt, res_tt))
    if norm > eps:
        res_tt = tt_scale(np.divide(2, norm), res_tt)
        A = tt_diag(res_tt)
        if degenerate:
            A = tt_add(A, tt_scale(2*eps, tt_identity(len(A))))
        A = tt_rank_reduce(A, op_tol, variable_error=True)
        return _tt_is_psd(A, nswp=nswp, eps=eps, verbose=verbose)
    return True, 0.0
