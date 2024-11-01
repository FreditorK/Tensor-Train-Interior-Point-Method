import sys
import os

sys.path.append(os.getcwd() + '/../')

import numpy as np
import scipy as scip
import time
from src.tt_ops import tt_ranks, tt_random_gaussian, tt_rank_retraction, tt_linear_op, tt_matrix_vec_mul, tt_sub


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
        print(f"\tTT-Matrix rank: {tt_ranks(A)}")
        print(f"\tTT-bias rank: {tt_ranks(b)}")

    for swp in range(nswp):
        for k in range(d - 1, 0, -1):
            # right to left orthogonalization
            if not last:
                if swp > 0:
                    # shape rzp x N x rz
                    czA = np.einsum('lsr,smnS,LSR,rnR->lmL', Phiz[k], A[k], Phiz[k + 1], x_cores[k], optimize=True)
                    # shape is rzp x N x rz
                    czy = np.einsum('br,bnB,BR->rnR', Phiz_b[k], b[k], Phiz_b[k + 1], optimize=True)
                    cz_new = czy * nrmsc - czA
                    _, _, vz = scip.linalg.svd(np.reshape(cz_new, [cz_new.shape[0], -1]), full_matrices=False)
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

            core_prev = np.einsum('ijk,km->ijm', x_cores[k - 1], Rmat.T, optimize=True)
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
            Phis[k] = np.einsum('LSR,lML,sMNS,rNR->lsr', Phis[k + 1], x_cores[k], A[k], x_cores[k], optimize=True)
            Phis_b[k] = np.einsum('BR,bnB,rnR->br', Phis_b[k + 1], b[k], x_cores[k], optimize=True)

            # ... and norms
            norm = np.linalg.norm(Phis[k])
            norm = norm if norm > 0 else 1.0
            normA[k - 1] = norm
            Phis[k] /= norm
            norm = np.linalg.norm(Phis_b[k])
            norm = norm if norm > 0 else 1.0
            normb[k - 1] = norm
            Phis_b[k] /= norm

            # norm correction
            nrmsc = nrmsc * normb[k - 1] / (normA[k - 1] * normx[k - 1])

            # compute phis_z
            Phiz[k] = np.einsum('LSR,lML,sMNS,rNR->lsr', Phiz[k + 1], z_cores[k], A[k], x_cores[k], optimize=True) / normA[k - 1]
            Phiz_b[k] = np.einsum('BR,bnB,rnR->br', Phiz_b[k + 1], b[k], z_cores[k], optimize=True) / normb[k - 1]

        # start loop
        max_res = 0

        for k in range(d):
            # assemble rhs
            rhs = np.einsum('br,bmB,BR->rmR', Phis_b[k], b[k] * nrmsc, Phis_b[k + 1], optimize=True)
            rhs = np.reshape(rhs, [-1, 1])
            norm_rhs = np.linalg.norm(rhs)

            # residuals
            real_tol = (eps / np.sqrt(d)) / damp

            # solve the local system
            # shape is Rp x N x N x r x r
            Bp = np.einsum('smnS,LSR->smnRL', A[k], Phis[k + 1], optimize=True)
            local_matrix_core = np.einsum('lsr,smnRL->lmLrnR', Phis[k], Bp, optimize=True)
            B = np.reshape(local_matrix_core, [rx[k] * N[k] * rx[k + 1], rx[k] * N[k] * rx[k + 1]])

            # We need the solution with the lowest x-norm so that the dual variables Y do not diverge
            # TODO: Prove that the regularisation yields minimum norm solution for Delta
            reg_B = B + reg_lambda * np.identity(B.shape[0])

            solution_now = scip.linalg.solve(reg_B, rhs, check_finite=False)

            res_new = np.linalg.norm(B @ solution_now - rhs) / norm_rhs

            max_res = max(max_res, res_new)

            solution_now = np.reshape(solution_now, (rx[k] * N[k], rx[k + 1]))
            # truncation
            if k < d - 1:
                tol = max(real_tol * damp, res_new)
                U, s, V = scip.linalg.svd(solution_now, full_matrices=False)
                V = np.diag(s) @ V
                for r in range(U.shape[1], 0, -1):
                    trunc_solution = U[:, :r] @ V[:r, :]
                    solution = trunc_solution.reshape((rx[k], N[k], rx[k + 1]))
                    local_core = np.einsum('lmLrnR,rnR->lmL', local_matrix_core, solution, optimize=True)
                    res = np.linalg.norm(local_core.reshape(*rhs.shape) - rhs)

                    if res > tol:
                        break

                r = min(r + 1, len(s))
                u = U[:, :r]
                v = V[:r, :].T
            else:
                u, v = np.linalg.qr(solution_now)
                v = v.T

            if not last:
                czA = np.einsum('lsr,smnS,LSR,rnR->lmL', Phiz[k], A[k], Phiz[k + 1], np.reshape(u @ v.T, [rx[k], N[k], rx[k + 1]]), optimize=True)

                # shape is rzp x N x rz
                czy = np.einsum('br,bnB,BR->rnR', Phiz_b[k], b[k] * nrmsc, Phiz_b[k + 1], optimize=True)
                cz_new = czy - czA
                uz, _, _ = scip.linalg.svd(np.reshape(cz_new, (rz[k] * N[k], -1)), full_matrices=False)
                # truncate to kickrank
                qz = uz[:, :min(kickrank, uz.shape[1])]
                rz[k + 1] = qz.shape[1]
                z_cores[k] = np.reshape(qz, (rz[k], N[k], rz[k + 1]))

            if k < d - 1:
                if not last:
                    left_res = np.einsum('lsr,smnS,LSR,rnR->lmL', Phis[k], A[k], Phiz[k + 1], np.reshape(u @ v.T, [rx[k], N[k], rx[k + 1]]), optimize=True)
                    left_b = np.einsum('br,bmB,BR->rmR', Phis_b[k], b[k] * nrmsc, Phiz_b[k + 1], optimize=True)
                    uk = left_b - left_res  # rx_k x N_k x rz_k+1
                    u, Rmat = np.linalg.qr(np.concatenate((u, np.reshape(uk, [u.shape[0], -1])), 1))
                    r_add = uk.shape[2]
                    v = np.concatenate((v, np.zeros([rx[k + 1], r_add], dtype=dtype)), 1)
                    v = v @ Rmat.T

                r = u.shape[1]
                v = np.einsum('ji,jkl->ikl', v, x_cores[k + 1], optimize=True)
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
                Phis[k + 1] = np.einsum('lsr,lML,sMNS,rNR->LSR', Phis[k], x_cores[k], A[k], x_cores[k], optimize=True)
                Phis_b[k + 1] = np.einsum('br,bnB,rnR->BR', Phis_b[k], b[k], x_cores[k], optimize=True)

                # ... and norms
                norm = np.linalg.norm(Phis[k + 1])
                norm = norm if norm > 0 else 1.0
                normA[k] = norm
                Phis[k + 1] /= norm
                norm = np.linalg.norm(Phis_b[k + 1])
                norm = norm if norm > 0 else 1.0
                normb[k] = norm
                Phis_b[k + 1] /= norm

                # norm correction
                nrmsc = nrmsc * normb[k] / (normA[k] * normx[k])

                # next phiz
                Phiz[k + 1] = np.einsum('lsr,lML,sMNS,rNR->LSR', Phiz[k], z_cores[k], A[k], x_cores[k], optimize=True) / normA[k]
                Phiz_b[k + 1] = np.einsum('br,bnB,rnR->BR', Phiz_b[k], b[k], z_cores[k], optimize=True) / normb[k]
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
        print('\tTime per sweep: ', (time.time() - t0)/swp)

    return x_cores, max_res
