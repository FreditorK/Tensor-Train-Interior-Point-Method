import sys
import os

sys.path.append(os.getcwd() + '/../')

import numpy as np
import scipy as scip
import time
from src.tt_ops import *
from src.tt_amen import compute_phi_bcks_A, compute_phi_fwd_A, solution_truncation
from cy_src.ops_cy import *



def tt_is_geq(matrix, x, b, nswp=10, eps=1e-10, verbose=False):
    if verbose:
        print(f"Starting Eigen solve with:\n \t {eps} \n \t sweeps: {nswp}")
        t0 = time.time()
    dtype = matrix[0].dtype
    damp = 2

    min_or_max = "SA"

    res_tt = tt_rl_orthogonalise(tt_sub(tt_matrix_vec_mul(matrix, x), b))
    A = tt_diag(res_tt)

    x_cores = tt_random_gaussian([1]*(len(A)-1), shape=(2, ))

    d = len(x_cores)
    rx = np.array([1] + tt_ranks(x_cores) + [1])
    N = np.array([c.shape[1] for c in x_cores])

    Phis = [np.ones((1, 1, 1), dtype=dtype)] + [None] * (d - 1) + [np.ones((1, 1, 1), dtype=dtype)]  # size is rk x Rk x rk

    max_res = 0
    for swp in range(nswp):
        x_cores = tt_rl_orthogonalise(x_cores)
        rx[1:-1] = np.array(tt_ranks(x_cores))
        Phis, _ = compute_phi_bcks_A(Phis, x_cores, A, x_cores, d=d)

        # start loop
        max_res = 0

        for k in range(d):

            # residuals
            real_tol = (eps / np.sqrt(d)) / damp

            # solve the local system
            Bp = np.einsum("smnS,LSR->smnRL", A[k], Phis[k + 1])
            B = np.einsum("lsr,smnRL->lmLrnR", Phis[k], Bp)
            B = np.reshape(B, [rx[k] * N[k] * rx[k + 1], rx[k] * N[k] * rx[k + 1]])

            eig_val, solution_now = scip.sparse.linalg.eigsh(B, k=1, which=min_or_max)
            rhs = eig_val * solution_now

            res_new = np.linalg.norm(B @ solution_now - rhs)

            max_res = max(max_res, res_new)

            solution_now = np.reshape(solution_now, (rx[k] * N[k], rx[k + 1]))
            # truncation
            if k < d - 1:
                u, v = solution_truncation(solution_now, Phis[k], Phis[k + 1], A[k], rhs, rx[k], N[k], rx[k + 1], max(real_tol * damp, res_new))
                r = u.shape[1]
            else:
                u, v = np.linalg.qr(solution_now)
                r = u.shape[1]
                v = v.T


            if k < d - 1:
                v = np.einsum('ji,jkl->ikl', v, x_cores[k + 1])

                x_cores[k] = np.reshape(u, [rx[k], N[k], r])
                x_cores[k + 1] = np.reshape(v, [r, N[k + 1], rx[k + 2]])
                rx[k + 1] = r

                # next phis with norm correction
                Phis[k + 1] = compute_phi_fwd_A(Phis[k], x_cores[k], A[k], x_cores[k])

                # ... and norms
                norm = np.linalg.norm(Phis[k + 1])
                norm = norm if np.greater(norm, 0) else 1.0
                Phis[k + 1] = np.divide(Phis[k + 1], norm)

            else:
                x_cores[k] = np.reshape(u @ v.T, (rx[k], N[k], rx[k + 1]))

        x_cores = tt_normalise(x_cores)

        if max_res < eps:
            break
    if verbose:
        print("\t -----")
        print(f"\t Solution rank is {rx[1:-1]}")
        print(f"\t Residual {max_res}")
        print(f"\t Time: {time.time() - t0:4f}s")

    x_cores = tt_rank_retraction(tt_matrix_vec_mul(A, x_cores), [1]*(d-1))
    i_cores = [np.squeeze(c) for c in x_cores]
    indices = [1 if c[0] > c[1] else 0 for c in i_cores]
    min_val = tt_entry(x_cores, indices)
    return np.greater_equal(min_val, 0), min_val
