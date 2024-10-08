import sys
import os

sys.path.append(os.getcwd() + '/../')

import numpy as np
import scipy as scip
import time
from src.tt_ops import tt_ranks, tt_normalise, tt_inner_prod, tt_matrix_vec_mul
from src.tt_amen import _compute_phi_bck_A, _compute_phi_fwd_A, _local_product


def tt_max_eig(matrix_tt, nswp=1, x0=None, eps=1e-10, trunc_norm='res', verbose=False):
    return _tt_eig(matrix_tt, min_eig=False, nswp=nswp, x0=x0, eps=eps, trunc_norm=trunc_norm, verbose=verbose)


def tt_min_eig(matrix_tt, nswp=1, x0=None, eps=1e-10, trunc_norm='res', verbose=False):
    return _tt_eig(matrix_tt, min_eig=True, nswp=nswp, x0=x0, eps=eps, trunc_norm=trunc_norm, verbose=verbose)


def _tt_eig(matrix_tt, min_eig=False, nswp=1, x0=None, eps=1e-10, trunc_norm='res', verbose=False):
    if verbose:
        print(f"Starting Eigen solve with:\n \t {eps} \n \t sweeps: {nswp}")
        t0 = time.time()
    dtype = matrix_tt[0].dtype
    damp = 2

    which = "SA" if min_eig else "LA"

    if x0 == None:
        x_cores = [np.ones_like(c[:, :, 0], dtype=dtype) for c in matrix_tt]
    else:
        x_cores = x0.copy()

    d = len(x_cores)
    rx = [1] + tt_ranks(x_cores) + [1]
    N = [c.shape[1] for c in x_cores]

    Phis = [np.ones((1, 1, 1), dtype=dtype)] + [None] * (d - 1) + [
        np.ones((1, 1, 1), dtype=dtype)]  # size is rk x Rk x rk

    normA = np.ones((d - 1))

    max_res = 0
    for swp in range(nswp):
        # right to left orthogonalization

        for k in range(d - 1, 0, -1):
            core = np.reshape(x_cores[k], [rx[k], N[k] * rx[k + 1]]).T
            Qmat, Rmat = np.linalg.qr(core)

            core_prev = np.einsum('ijk,km->ijm', x_cores[k - 1], Rmat.T)
            rx[k] = Qmat.shape[1]

            x_cores[k] = np.reshape(Qmat.T, (rx[k], N[k], rx[k + 1]))
            x_cores[k - 1] = core_prev[:]

            # update phis (einsum)
            Phis[k] = _compute_phi_bck_A(Phis[k + 1], x_cores[k], matrix_tt[k], x_cores[k])

            # ... and norms
            norm = np.linalg.norm(Phis[k])
            norm = norm if np.greater(norm, 0) else 1.0
            normA[k - 1] = norm
            Phis[k] = np.divide(Phis[k], norm)

        # start loop
        max_res = 0

        for k in range(d):

            # residuals
            real_tol = (eps / np.sqrt(d)) / damp

            # solve the local system
            # shape is Rp x N x N x r x r
            Bp = np.einsum("smnS,LSR->smnRL", matrix_tt[k], Phis[k + 1])
            B = np.einsum("lsr,smnRL->lmLrnR", Phis[k], Bp)
            B = np.reshape(B, [rx[k] * N[k] * rx[k + 1], rx[k] * N[k] * rx[k + 1]])

            eig_val, solution_now = scip.sparse.linalg.eigsh(B, k=1, which=which)
            rhs = eig_val * solution_now

            res_new = np.linalg.norm(B @ solution_now - rhs)

            max_res = max(max_res, res_new)

            solution_now = np.reshape(solution_now, (rx[k] * N[k], rx[k + 1]))
            # truncation
            if k < d - 1:
                u, s, v = np.linalg.svd(solution_now, full_matrices=False)
                if trunc_norm != 'fro':
                    # search for a rank such that offers small enough residuum
                    r = 0
                    for r in range(u.shape[1] - 1, 0, -1):
                        # solution has the same size
                        solution = u[:, :r] @ np.diag(s[:r]) @ v[:r, :]
                        res = np.linalg.norm(np.reshape(
                            _local_product(Phis[k + 1], Phis[k], matrix_tt[k], np.reshape(solution, [rx[k], N[k], rx[k + 1]]),
                                           solution_now.shape), [-1, 1]) - rhs)

                        if res > max(real_tol * damp, res_new):
                            break
                    r = min(r + 1, np.prod(s.shape))
            else:
                u, v = np.linalg.qr(solution_now)
                r = u.shape[1]
                s = np.ones(r, dtype=dtype)

            u = u[:, :r]
            v = np.diag(s[:r]) @ v[:r, :]
            v = v.T

            if k < d - 1:

                r = u.shape[1]
                v = np.einsum('ji,jkl->ikl', v, x_cores[k + 1])

                x_cores[k] = np.reshape(u, [rx[k], N[k], r])
                x_cores[k + 1] = np.reshape(v, [r, N[k + 1], rx[k + 2]])
                rx[k + 1] = r

                # next phis with norm correction
                Phis[k + 1] = _compute_phi_fwd_A(Phis[k], x_cores[k], matrix_tt[k], x_cores[k])

                # ... and norms
                norm = np.linalg.norm(Phis[k + 1])
                norm = norm if np.greater(norm, 0) else 1.0
                normA[k] = norm
                Phis[k + 1] = np.divide(Phis[k + 1], norm)

            else:
                x_cores[k] = np.reshape(u @ np.diag(s[:r]) @ v[:r, :].T, (rx[k], N[k], rx[k + 1]))

        x_cores = tt_normalise([x_cores[k] for k in range(d)])

        if max_res < eps:
            break
    if verbose:
        print("\t -----")
        print(f"\t Solution rank is {rx[1:-1]}")
        print(f"\t Residual {max_res}")
        print(f"\t Time: {time.time() - t0:4f}s")
    return tt_inner_prod(x_cores, tt_matrix_vec_mul(matrix_tt, x_cores)), x_cores, max_res
