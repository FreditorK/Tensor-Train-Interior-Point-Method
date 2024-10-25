import sys
import os

sys.path.append(os.getcwd() + '/../')

import numpy as np
import scipy as scip
import time
from src.tt_ops import tt_ranks, tt_random_gaussian, tt_rl_orthogonalise, _tt_rl_random_orthogonalise
from cy_src.ops_cy import *


def tt_amen(A, b, nswp=5, x0=None, eps=1e-10, kickrank=2, reg_lambda=1e-12, verbose=False):
    if verbose:
        t0 = time.time()

    dtype = A[0].dtype
    damp = 2

    if x0 == None:
        x = [np.ones_like(c, dtype=dtype) for c in b]
    else:
        x = x0

    d = len(x)
    x_cores = x.copy()
    rx = [1] + tt_ranks(x) + [1]
    N = [c.shape[1] for c in x_cores]

    # z cores
    z_cores = tt_random_gaussian((d - 1) * [kickrank], shape=x_cores[0].shape[1:-1])
    rz = [1] + tt_ranks(z_cores) + [1]

    Phiz = [np.ones((1, 1, 1), dtype=dtype)] + [None] * (d - 1) + [
        np.ones((1, 1, 1), dtype=dtype)]  # size is rzk x Rk x rxk
    Phiz_b = [np.ones((1, 1), dtype=dtype)] + [None] * (d - 1) + [np.ones((1, 1), dtype=dtype)]  # size is rzk x rzbk
    Phis = [np.ones((1, 1, 1), dtype=dtype)] + [None] * (d - 1) + [
        np.ones((1, 1, 1), dtype=dtype)]  # size is rk x Rk x rk
    Phis_b = [np.ones((1, 1), dtype=dtype)] + [None] * (d - 1) + [np.ones((1, 1), dtype=dtype)]  # size is rk x rbk

    last = False

    # TODO: The SVDs of z-related stuff to kickrank can be done with randomised orthogonolise

    normA = np.ones((d - 1))
    normb = np.ones((d - 1))
    normx = np.ones((d - 1))
    nrmsc = 1.0

    if verbose:
        print('Starting AMEn solve with:\n\tepsilon: %g\n\tsweeps: %d' % (eps, nswp))

    for swp in range(nswp):
        for k in range(d - 1, 0, -1):
            # right to left orthogonalization
            if not last:
                if swp > 0:
                    # shape rzp x N x rz
                    czA = local_product(Phiz[k + 1], Phiz[k], A[k], x_cores[k])
                    # shape is rzp x N x rz
                    czy = np.einsum('br,bnB,BR->rnR', Phiz_b[k], b[k], Phiz_b[k + 1])
                    cz_new = czy * nrmsc - czA
                    _, _, vz = np.linalg.svd(np.reshape(cz_new, [cz_new.shape[0], -1]), full_matrices=False)
                    # truncate to kickrank
                    cz_new = vz[:min(kickrank, vz.shape[0]), :].T
                else:
                    cz_new = np.reshape(z_cores[k], (rz[k], -1)).T

                qz, _ = np.linalg.qr(cz_new)
                rz[k] = qz.shape[1]
                z_cores[k] = np.reshape(qz.T, (rz[k], N[k], -1))

            # norm correction ?
            if swp > 0:
                nrmsc = nrmsc * normA[k - 1] * normx[k - 1] / normb[k - 1]

            core = np.reshape(x_cores[k], [rx[k], N[k] * rx[k + 1]]).T
            Qmat, Rmat = np.linalg.qr(core)

            core_prev = np.einsum('ijk,km->ijm', x_cores[k - 1], Rmat.T)
            rx[k] = Qmat.shape[1]

            current_norm = np.linalg.norm(core_prev)
            if current_norm > 0:
                core_prev = core_prev / current_norm
            else:
                current_norm = 1.0
            normx[k - 1] = normx[k - 1] * current_norm

            x_cores[k] = np.reshape(Qmat.T, (rx[k], N[k], rx[k + 1]))
            x_cores[k - 1] = core_prev[:]

            # update phis (einsum)
            Phis[k] = compute_phi_bck_A(Phis[k + 1], x_cores[k], A[k], x_cores[k])
            Phis_b[k] = compute_phi_bck_rhs(Phis_b[k + 1], b[k], x_cores[k])

            # ... and norms
            norm = np.linalg.norm(Phis[k])
            norm = norm if norm > 0 else 1.0
            normA[k - 1] = norm
            Phis[k] = Phis[k] / norm
            norm = np.linalg.norm(Phis_b[k])
            norm = norm if norm > 0 else 1.0
            normb[k - 1] = norm
            Phis_b[k] = Phis_b[k] / norm

            # norm correction
            nrmsc = nrmsc * normb[k - 1] / (normA[k - 1] * normx[k - 1])

            # compute phis_z
            Phiz[k] = compute_phi_bck_A(Phiz[k + 1], z_cores[k], A[k], x_cores[k]) / normA[k - 1]
            Phiz_b[k] = compute_phi_bck_rhs(Phiz_b[k + 1], b[k], z_cores[k]) / normb[k - 1]

        # start loop
        max_res = 0

        for k in range(d):
            # assemble rhs
            rhs = np.einsum('br,bmB,BR->rmR', Phis_b[k], b[k] * nrmsc, Phis_b[k + 1])
            rhs = np.reshape(rhs, [-1, 1])
            norm_rhs = np.linalg.norm(rhs)

            # residuals
            real_tol = (eps / np.sqrt(d)) / damp

            # solve the local system
            # shape is Rp x N x N x r x r
            Bp = np.einsum('smnS,LSR->smnRL', A[k], Phis[k + 1])
            B = np.einsum('lsr,smnRL->lmLrnR', Phis[k], Bp)
            B = np.reshape(B, [rx[k] * N[k] * rx[k + 1], rx[k] * N[k] * rx[k + 1]])

            # We need the solution with the lowest x-norm so that the dual variables Y do not diverge
            # TODO: Prove that the regularisation yields minimum norm solution for Delta
            reg_B = B + reg_lambda * np.identity(B.shape[0])
            solution_now, _, _, _ = scip.linalg.lstsq(reg_B, rhs, cond=1e-18, check_finite=False)

            res_new = np.linalg.norm(B @ solution_now - rhs) / norm_rhs

            max_res = max(max_res, res_new)

            solution_now = np.reshape(solution_now, (rx[k] * N[k], rx[k + 1]))
            # truncation
            if k < d - 1:
                u, v = solution_truncation(solution_now, Phis[k], Phis[k + 1], A[k], rhs, rx[k], N[k], rx[k + 1],
                                           max(real_tol * damp, res_new))
            else:
                u, v = np.linalg.qr(solution_now)
                v = v.T

            if not last:
                czA = local_product(
                    Phiz[k + 1],
                    Phiz[k],
                    A[k],
                    np.reshape(u @ v.T, [rx[k], N[k], rx[k + 1]])
                )  # shape rzp x N x rz

                # shape is rzp x N x rz
                czy = np.einsum('br,bnB,BR->rnR', Phiz_b[k], b[k] * nrmsc, Phiz_b[k + 1])
                cz_new = czy - czA
                uz, _, _ = np.linalg.svd(np.reshape(cz_new, (rz[k] * N[k], -1)), full_matrices=False)
                # truncate to kickrank
                qz = uz[:, :min(kickrank, uz.shape[1])]
                rz[k + 1] = qz.shape[1]
                z_cores[k] = np.reshape(qz, (rz[k], N[k], rz[k + 1]))

            if k < d - 1:
                if not last:
                    left_res = local_product(Phiz[k + 1], Phis[k], A[k], np.reshape(
                        u @ v.T, [rx[k], N[k], rx[k + 1]]))
                    left_b = np.einsum('br,bmB,BR->rmR', Phis_b[k], b[k] * nrmsc, Phiz_b[k + 1])
                    uk = left_b - left_res  # rx_k x N_k x rz_k+1
                    u, Rmat = np.linalg.qr(np.concatenate((u, np.reshape(uk, [u.shape[0], -1])), 1))
                    r_add = uk.shape[2]
                    v = np.concatenate((v, np.zeros([rx[k + 1], r_add], dtype=dtype)), 1)
                    v = v @ Rmat.T

                r = u.shape[1]
                v = np.einsum('ji,jkl->ikl', v, x_cores[k + 1])
                # remove norm correction
                nrmsc = nrmsc * normA[k] * normx[k] / normb[k]

                norm_now = np.linalg.norm(v)

                if norm_now > 0:
                    v = v / norm_now
                else:
                    norm_now = 1.0
                normx[k] = normx[k] * norm_now

                x_cores[k] = np.reshape(u, [rx[k], N[k], r])
                x_cores[k + 1] = np.reshape(v, [r, N[k + 1], rx[k + 2]])
                rx[k + 1] = r

                # next phis with norm correction
                Phis[k + 1] = compute_phi_fwd_A(Phis[k], x_cores[k], A[k], x_cores[k])
                Phis_b[k + 1] = compute_phi_fwd_rhs(Phis_b[k], b[k], x_cores[k])

                # ... and norms
                norm = np.linalg.norm(Phis[k + 1])
                norm = norm if norm > 0 else 1.0
                normA[k] = norm
                Phis[k + 1] = Phis[k + 1] / norm
                norm = np.linalg.norm(Phis_b[k + 1])
                norm = norm if norm > 0 else 1.0
                normb[k] = norm
                Phis_b[k + 1] = Phis_b[k + 1] / norm

                # norm correction
                nrmsc = nrmsc * normb[k] / (normA[k] * normx[k])

                # next phiz
                Phiz[k + 1] = compute_phi_fwd_A(Phiz[k], z_cores[k],
                                                 A[k], x_cores[k]) / normA[k]
                Phiz_b[k + 1] = compute_phi_fwd_rhs(
                    Phiz_b[k], b[k], z_cores[k]) / normb[k]
            else:
                x_cores[k] = np.reshape(u @ v.T, (rx[k], N[k], rx[k + 1]))

        if last:
            break
        if np.less(max_res, eps):
            last = True

    normx = np.exp(np.sum(np.log(normx)) / d)

    x_cores = [x_cores[k] * normx for k in range(d)]
    if verbose:
        print("\t---Results---")
        print('\tSolution rank is', rx[1:-1])
        print('\tResidual ', max_res)
        print('\tNumber of sweeps', swp)
        print('\tTime: ', time.time() - t0)

    return x_cores, max_res
