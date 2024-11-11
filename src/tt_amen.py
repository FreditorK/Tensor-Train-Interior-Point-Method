import sys
import os

import scipy.linalg

sys.path.append(os.getcwd() + '/../')

import numpy as np
import datetime
import opt_einsum as oe
from src.tt_ops import *
import time


def _local_product(Phi_right, Phi_left, coreA, core):

    w = oe.contract('lsr,smnS,LSR,rnR->lmL', Phi_left, coreA, Phi_right, core)
    return w


def tt_amen(A, b, nswp=22, x0=None, eps=1e-10, rmax=1024, max_full=500, kickrank=4, local_iterations=40, resets=2, verbose=False):

    dtype = A[0].dtype
    damp = 2

    if x0 == None:
        x = [np.ones_like(c, dtype=dtype) for c in b]
    else:
        x = x0

    x_cores = copy.copy(x)
    N = [c.shape[1] for c in x_cores]
    d = len(N)
    rx = [1] + tt_ranks(x) + [1]
    rmax = [1] + (d - 1) * [rmax] + [1]

    # z cores
    z_tt = tt_random_gaussian([kickrank]*(d - 1), shape=x_cores[0].shape)
    z_tt = tt_rl_random_orthogonalise(z_tt, (d - 1) * [kickrank])
    rz = [1] + tt_ranks(z_tt) + [1]

    Phiz = [np.ones((1, 1, 1), dtype=dtype)] + [None] * (d - 1) + [np.ones((1, 1, 1), dtype=dtype)]  # size is rzk x Rk x rxk
    Phiz_b = [np.ones((1, 1), dtype=dtype)] + [None] * (d - 1) + [np.ones((1, 1), dtype=dtype)]  # size is rzk x rzbk

    Phis = [np.ones((1, 1, 1), dtype=dtype)] + [None] * (d - 1) + [np.ones((1, 1, 1), dtype=dtype)]  # size is rk x Rk x rk
    Phis_b = [np.ones((1, 1), dtype=dtype)] + [None] * (d - 1) + [np.ones((1, 1), dtype=dtype)]  # size is rk x rbk

    last = False

    normA = np.ones((d - 1))
    normb = np.ones((d - 1))
    normx = np.ones((d - 1))
    nrmsc = 1.0

    if verbose:
        t0 = time.time()
        print('Starting AMEn solve with:\n\tepsilon: %g\n\tsweeps: %d' % (eps, nswp))
        print(f"\tTT-Matrix rank: {tt_ranks(A)}")
        print(f"\tTT-bias rank: {tt_ranks(b)}")

    for swp in range(nswp):
        # right to left orthogonalization

        if verbose:
            print()
            print('Starting sweep %d %s...' %
                  (swp + 1, "(last one) " if last else ""))
            tme_sweep = datetime.datetime.now()

        for k in range(d - 1, 0, -1):

            # update the z part (ALS) update
            if not last:
                if swp > 0:
                    # shape rzp x N x rz
                    czA = _local_product(Phiz[k + 1], Phiz[k], A[k], x_cores[k])
                    # shape is rzp x N x rz
                    czy = np.einsum('br,bnB,BR->rnR',
                                    Phiz_b[k], b[k], Phiz_b[k + 1])
                    cz_new = czy * nrmsc - czA
                    _, _, vz = scip.linalg.svd(np.reshape(cz_new, [cz_new.shape[0], -1]))
                    # truncate to kickrank
                    cz_new = vz[:min(kickrank, vz.shape[0]), :].T
                else:
                    cz_new = np.reshape(z_tt[k], [rz[k], -1]).T

                # FIXME: Change from here...
                qz, _ = np.linalg.qr(cz_new)
                rz[k] = qz.shape[0]
                z_tt[k] = np.reshape(qz.T, [-1, N[k], rz[k]])
                # FIXME: To here

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

            x_cores[k] = np.reshape(Qmat.T, [rx[k], N[k], rx[k + 1]])
            x_cores[k - 1] = core_prev[:]

            # update phis (einsum)
            # print(x_cores[k].shape,A.cores[k].shape,x_cores[k].shape)
            Phis[k] = _compute_phi_bck_A(
                Phis[k + 1], x_cores[k], A[k], x_cores[k])
            Phis_b[k] = _compute_phi_bck_rhs(
                Phis_b[k + 1], b[k], x_cores[k])

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
            if not last:
                Phiz[k] = _compute_phi_bck_A(
                    Phiz[k + 1], z_tt[k], A[k], x_cores[k]) / normA[k - 1]
                Phiz_b[k] = _compute_phi_bck_rhs(
                    Phiz_b[k + 1], b[k], z_tt[k]) / normb[k - 1]

        # start loop
        max_res = 0
        max_dx = 0

        rz = [1] + tt_ranks(z_tt) + [1]

        for k in range(d):
            if verbose:
                print('\tCore', k)
            previous_solution = np.reshape(x_cores[k], [-1, 1])

            # assemble rhs
            rhs = np.einsum('br,bmB,BR->rmR',
                            Phis_b[k], b[k] * nrmsc, Phis_b[k + 1])
            rhs = np.reshape(rhs, [-1, 1])
            norm_rhs = np.linalg.norm(rhs)

            # residuals
            real_tol = (eps / np.sqrt(d)) / damp

            # solve the local system
            use_full = rx[k] * N[k] * rx[k + 1] < max_full
            # solve the full system
            if verbose:
                print('\t\tChoosing direct solver (local size %d)....' %
                      (rx[k] * N[k] * rx[k + 1]))
            # shape is Rp x N x N x r x r
            Bp = np.einsum('smnS,LSR->smnRL', A[k], Phis[k + 1])
            B = np.einsum('lsr,smnRL->lmLrnR', Phis[k], Bp)
            B = np.reshape(B, [rx[k] * N[k] * rx[k + 1], rx[k] * N[k] * rx[k + 1]])

            solution_now = np.linalg.solve(B, rhs)

            res_old = np.linalg.norm(B @ previous_solution - rhs) / norm_rhs
            res_new = np.linalg.norm(B @ solution_now - rhs) / norm_rhs

            # residual damp check
            if res_old / res_new < damp and res_new > real_tol:
                if verbose:
                    print('WARNING: residual increases. res_old %g, res_new %g, real_tol %g' % (
                        res_old, res_new, real_tol))  # warning (from tt toolbox)

            # compute residual and step size
            dx = np.linalg.norm(solution_now - previous_solution) / \
                 np.linalg.norm(solution_now)
            if verbose:
                print('\t\tdx = %g, res_now = %g, res_old = %g' %
                      (dx, res_new, res_old))

            max_dx = max(dx, max_dx)
            max_res = max(max_res, res_old)

            solution_now = np.reshape(solution_now, [rx[k] * N[k], rx[k + 1]])
            # truncation
            if k < d - 1:
                u, s, v = scip.linalg.svd(solution_now, full_matrices=False)
                r = 0
                print(u.shape, s.shape, v.shape, r)
                for r in range(u.shape[1] - 1, 0, -1):
                    # solution has the same size
                    print( u[:, :r].shape, v[:r, :].shape, r)
                    solution = u[:, :r] @ np.diag(s[:r]) @ v[:r, :]
                    res = np.linalg.norm(B @ np.reshape(solution, [-1, 1]) - rhs) / norm_rhs
                    if res > max(real_tol * damp, res_new):
                        break
                r += 1

                r = min([r, np.size(s), rmax[k + 1]])
            else:
                u, v = np.linalg.qr(solution_now)
                r = u.shape[1]
                s = np.ones(r, dtype=dtype)

            u = u[:, :r]
            v = np.diag(s[:r]) @ v[:r, :]
            v = v.T

            if not last:
                czA = _local_product(Phiz[k + 1], Phiz[k], A[k], np.reshape(u @ v.T, [rx[k], N[k], rx[k + 1]]))  # shape rzp x N x rz

                # shape is rzp x N x rz
                czy = np.einsum('br,bnB,BR->rnR', Phiz_b[k], b[k] * nrmsc, Phiz_b[k + 1])
                cz_new = czy - czA
                print(cz_new.shape, [rz[k] * N[k], rz[k + 1]])
                uz, _, _ = scipy.linalg.svd(np.reshape(cz_new, [rz[k] * N[k], rz[k + 1]]))
                # truncate to kickrank
                cz_new = uz[:, :min(kickrank, uz.shape[1])]

                qz, _ = np.linalg.qr(cz_new)
                rz[k + 1] = qz.shape[1]
                z_tt[k] = np.reshape(qz, [rz[k], N[k], rz[k + 1]])

            if k < d - 1:
                if not last:
                    left_res = _local_product(Phiz[k + 1], Phis[k], A[k], np.reshape(u @ v.T, [rx[k], N[k], rx[k + 1]]))
                    left_b =np.einsum(
                        'br,bmB,BR->rmR', Phis_b[k], b[k] * nrmsc, Phiz_b[k + 1])
                    uk = left_b - left_res  # rx_k x N_k x rz_k+1
                    u, Rmat = scip.linalg.qr(np.concatenate((u, np.reshape(uk, [u.shape[0], -1])), 1))
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
                Phis[k + 1] = _compute_phi_fwd_A(Phis[k],
                                                 x_cores[k], A[k], x_cores[k])
                Phis_b[k +
                       1] = _compute_phi_fwd_rhs(Phis_b[k], b[k], x_cores[k])

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
                if not last:
                    Phiz[k + 1] = _compute_phi_fwd_A(Phiz[k], z_tt[k],
                                                     A[k], x_cores[k]) / normA[k]
                    Phiz_b[k + 1] = _compute_phi_fwd_rhs(
                        Phiz_b[k], b[k], z_tt[k]) / normb[k]
            else:
                x_cores[k] = np.reshape(
                    u @ np.diag(s[:r]) @ v[:r, :].T, [rx[k], N[k], rx[k + 1]])

        if verbose:
            print('Solution rank is', rx)
            print('Maxres ', max_res)
            tme_sweep = datetime.datetime.now() - tme_sweep
            print('Time ', tme_sweep)

        if last:
            break

        if max_res < eps:
            last = True

    if verbose:
        print("\t---Results---")
        print('\tSolution rank is', rx[1:-1])
        print('\tResidual ', max_res)
        print('\tNumber of sweeps', swp)
        print('\tTime: ', time.time() - t0)
        print('\tTime per sweep: ', (time.time() - t0) / swp)

    normx = np.exp(np.sum(np.log(normx)) / d)

    for k in range(d):
        x_cores[k] = x_cores[k] * normx

    return x_cores


def _compute_phi_bck_A(Phi_now, core_left, core_A, core_right):
    Phi = oe.contract('LSR,lML,sMNS,rNR->lsr', Phi_now,
                      core_left, core_A, core_right)
    return Phi


def _compute_phi_fwd_A(Phi_now, core_left, core_A, core_right):
    Phi_next = oe.contract('lsr,lML,sMNS,rNR->LSR',
                           Phi_now, core_left, core_A, core_right)
    return Phi_next


def _compute_phi_bck_rhs(Phi_now, core_b, core):
    Phi = oe.contract('BR,bnB,rnR->br', Phi_now, core_b, core)
    return Phi


def _compute_phi_fwd_rhs(Phi_now, core_rhs, core):
    Phi_next = oe.contract('br,bnB,rnR->BR', Phi_now, core_rhs, core)
    return Phi_next

