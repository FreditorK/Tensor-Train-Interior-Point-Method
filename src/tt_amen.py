import sys
import os

from sympy.stats.sampling.sample_scipy import scipy

sys.path.append(os.getcwd() + '/../')

import numpy as np
import scipy as scip
import time
from src.tt_ops import tt_ranks, tt_random_gaussian, tt_rl_orthogonalise, tt_mat_mat_mul, tt_inner_prod, tt_sub, \
    tt_matrix_vec_mul
from cy_src.ops_cy import *


def tt_amen(A, b, nswp=5, x0=None, eps=1e-10, kickrank=2, kick2=0, verbose=False, band_diagonal=-1):
    if verbose:
        print('Starting AMEn solve with:\n\tepsilon: %g\n\tsweeps: %d' % (eps, nswp))
        t0 = time.time()

    dtype = A[0].dtype
    damp = 2

    if x0 == None:
        x = [np.ones_like(c, dtype=dtype) for c in b]
    else:
        x = x0

    d = len(x)
    x_cores = x.copy()
    x_cores = tt_rl_orthogonalise(x_cores)
    rx = [1] + tt_ranks(x) + [1]
    N = [c.shape[1] for c in x_cores]

    # z cores
    z_cores = tt_random_gaussian((d-1)*[kickrank+kick2], shape=x_cores[0].shape[1:-1])
    z_cores = tt_rl_orthogonalise(z_cores)
    rz = [1] + (d-1)*[kickrank+kick2] + [1]

    Phiz = [np.ones((1, 1, 1), dtype=dtype)] + [None] * (d-1) + [np.ones((1, 1, 1), dtype=dtype)]  # size is rzk x Rk x rxk
    Phiz_b = [np.ones((1, 1), dtype=dtype)] + [None] * (d-1) + [np.ones((1, 1), dtype=dtype)]   # size is rzk x rzbk
    Phis = [np.ones((1, 1, 1), dtype=dtype)] + [None] * (d-1) + [np.ones((1, 1, 1), dtype=dtype)]  # size is rk x Rk x rk
    Phis_b = [np.ones((1, 1), dtype=dtype)] + [None] * (d-1) + [np.ones((1, 1), dtype=dtype)]  # size is rk x rbk

    normx = np.ones((d-1))

    for swp in range(nswp):
        normx = np.cumprod([np.linalg.norm(np.reshape(x_cores[k], [rx[k], N[k]*rx[k+1]]).T) for k in range(d-1, 0, -1)])[::-1]

        Phis, normA = compute_phi_bcks_A(Phis, x_cores, A, x_cores, d=d)
        Phis_b, normb = compute_phi_bcks_rhs(Phis_b, b, x_cores, d=d)
        Phiz_b, _ = compute_phi_bcks_rhs(Phiz_b, b, z_cores, d=d)
        nrmsc = np.prod(normb / (normA * normx))
        Phiz, _ = compute_phi_bcks_A(Phiz, z_cores, A, x_cores, d=d)
        # start loop
        max_res = 0

        for k in range(d):
            # assemble rhs
            rhs = np.einsum('br,bmB,BR->rmR',Phis_b[k], b[k] * nrmsc, Phis_b[k+1])
            rhs = np.reshape(rhs, [-1, 1])
            norm_rhs = np.linalg.norm(rhs)

            # residuals
            real_tol = (eps/np.sqrt(d))/damp

            # solve the local system
            Bp = np.einsum('smnS,LSR->smnRL', A[k], Phis[k+1])
            B = np.einsum('lsr,smnRL->lmLrnR', Phis[k], Bp)
            B = np.reshape(B, [rx[k]*N[k]*rx[k+1], rx[k]*N[k]*rx[k+1]])

            solution_now, _, _, _ = scip.linalg.lstsq(B, rhs, cond=1e-18, check_finite=False)
            res_new = np.linalg.norm(B @ solution_now - rhs)/norm_rhs
            max_res = max(max_res, res_new)

            solution_now = np.reshape(solution_now, (rx[k]*N[k], rx[k+1]))
            # truncation
            if k < d - 1:
                u, v = solution_truncation(solution_now, Phis[k], Phis[k + 1], A[k], rhs, rx[k], N[k], rx[k + 1], max(real_tol * damp, res_new))
            else:
                u, v = np.linalg.qr(solution_now)
                v = v.T

            czA = local_product(
                Phiz[k+1],
                Phiz[k],
                A[k],
                np.reshape(u @ v.T, [rx[k], N[k], rx[k+1]]),
                band_diagonal
            )  # shape rzp x N x rz

            # shape is rzp x N x rz
            czy = np.einsum('br,bnB,BR->rnR',Phiz_b[k], b[k]*nrmsc, Phiz_b[k+1])
            cz_new = czy - czA
            uz, _ = np.linalg.qr(np.reshape(cz_new, (rz[k]*N[k], -1)))
            # truncate to kickrank
            cz_new = uz[:, :min(kickrank, uz.shape[1])]
            if k < d-1 and kick2 > 0:  # extend cz_new with random elements
                cz_new = np.concatenate((cz_new, np.random.randn(cz_new.shape[0], kick2)), 1)
                cz_new, _ = np.linalg.qr(cz_new)

            rz[k+1] = cz_new.shape[1]
            z_cores[k] = np.reshape(cz_new, (rz[k], N[k], rz[k+1]))

            if k < d-1:
                left_res = local_product(Phiz[k + 1], Phis[k], A[k], np.reshape(
                    u @ v.T, [rx[k], N[k], rx[k+1]]), band_diagonal)
                left_b = np.einsum('br,bmB,BR->rmR', Phis_b[k], b[k]*nrmsc, Phiz_b[k+1])
                uk = left_b - left_res  # rx_k x N_k x rz_k+1
                u, Rmat = np.linalg.qr(np.concatenate((u, np.reshape(uk, [u.shape[0], -1])), 1))
                r_add = uk.shape[2]
                v = np.concatenate((v, np.zeros([rx[k+1], r_add],  dtype=dtype)), 1)
                v = v @ Rmat.T

                r = u.shape[1]
                v = np.einsum('ji,jkl->ikl', v, x_cores[k+1])
                # remove norm correction
                nrmsc = nrmsc * normA[k] * normx[k] / normb[k]

                norm_now = np.linalg.norm(v)

                if norm_now > 0:
                    v = v / norm_now
                else:
                    norm_now = 1.0
                normx[k] = normx[k] * norm_now

                x_cores[k] = np.reshape(u, [rx[k], N[k], r])
                x_cores[k+1] = np.reshape(v, [r, N[k+1], rx[k+2]])
                rx[k+1] = r

                # next phis with norm correction
                Phis[k+1] = compute_phi_fwd_A(Phis[k], x_cores[k], A[k], x_cores[k])
                Phis_b[k+1] = compute_phi_fwd_rhs(Phis_b[k], b[k], x_cores[k])

                # ... and norms
                norm = np.linalg.norm(Phis[k+1])
                norm = norm if norm > 0 else 1.0
                normA[k] = norm
                Phis[k+1] = Phis[k+1] / norm
                norm = np.linalg.norm(Phis_b[k+1])
                norm = norm if norm > 0 else 1.0
                normb[k] = norm
                Phis_b[k+1] = Phis_b[k+1] / norm

                # norm correction
                nrmsc = nrmsc * normb[k] / (normA[k] * normx[k])

                # next phiz
                Phiz[k+1] = compute_phi_fwd_A(Phiz[k], z_cores[k],
                                              A[k], x_cores[k]) / normA[k]
                Phiz_b[k+1] = compute_phi_fwd_rhs(
                    Phiz_b[k], b[k], z_cores[k]) / normb[k]
            else:
                x_cores[k] = np.reshape(u @ v.T, (rx[k], N[k], rx[k+1]))

        if np.less(max_res, eps):
            break

    normx = np.exp(np.sum(np.log(normx))/d)

    x_cores = [x_cores[k] * normx for k in range(d)]
    if verbose:
        print("\t-----")
        print(f"\tSolution rank is {rx[1:-1]}")
        print(f"\tResidual {max_res}")
        print(f"\tSweeps taken {swp+1}")
        print(f"\tTime taken {time.time() - t0:4f}s")

    return x_cores, max_res
