import sys
import os

from sympy.stats.sampling.sample_scipy import scipy

sys.path.append(os.getcwd() + '/../')

import numpy as np
import scipy as scip
import time
import datetime
from src.tt_ops import *

def _local_product(Phi_right, Phi_left, coreA, core, shape, bandA=-1):

    if bandA < 0:
        w = np.einsum('lsr,smnS,LSR,rnR->lmL', Phi_left, coreA, Phi_right, core)
    else:
        w = 0
        for i in range(-bandA, bandA+1):
            tmp = np.diagonal(coreA, i, 1, 2)
            tmp = np.pad(tmp, ((i) if i > 0 else 0,
                          abs(i) if i < 0 else 0, 0, 0, 0, 0))
            tmp = np.einsum('lsr,sSm,LSR,rmR->lmL', Phi_left, tmp, Phi_right, core)
            if i < 0:
                tmp = np.pad(tmp[:, :i, :], (0, 0, -i, 0, 0, 0))
            else:
                tmp = np.pad(tmp[:, i:, :], (0, 0, 0, i, 0, 0))
            w += tmp
    return w


def rank_chop(s, eps):
    if np.linalg.norm(s) == 0.0:
        return 1

    if eps <= 0.0:
        return s.size

    sc = np.cumsum(np.abs(s[::-1]) ** 2)[::-1]
    R = np.argmax(sc < eps ** 2)
    R = R if R > 0 else 1
    R = s.size if sc[-1] > eps ** 2 else R

    return R


def tt_amen(A, b, nswp=22, x0=None, eps=1e-10, kickrank=4, kick2=0, trunc_norm='res', verbose=False, band_diagonal=-1):

    return _amen_solve_python(A, b, nswp, x0, eps, kickrank, kick2, trunc_norm, verbose, band_diagonal)


def _amen_solve_python(A, b, nswp=22, x0=None, eps=1e-10, kickrank=4, kick2=0, trunc_norm='res', verbose=False, band_diagonal=-1):
    if verbose:
        time_total = datetime.datetime.now()

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
    z_cores = tt_random_gaussian((d-1)*[kickrank+kick2], shape=x_cores[0].shape[1:-1])
    z_cores = tt_rl_orthogonalise(z_cores)
    rz = [1] + tt_ranks(z_cores) + [1]

    Phiz = [np.ones((1, 1, 1), dtype=dtype)] + [None] * (d-1) + [np.ones((1, 1, 1), dtype=dtype)]  # size is rzk x Rk x rxk
    Phiz_b = [np.ones((1, 1), dtype=dtype)] + [None] * (d-1) + [np.ones((1, 1), dtype=dtype)]   # size is rzk x rzbk
    Phis = [np.ones((1, 1, 1), dtype=dtype)] + [None] * (d-1) + [np.ones((1, 1, 1), dtype=dtype)]  # size is rk x Rk x rk
    Phis_b = [np.ones((1, 1), dtype=dtype)] + [None] * (d-1) + [np.ones((1, 1), dtype=dtype)]  # size is rk x rbk

    last = False

    normA = np.ones((d-1))
    normb = np.ones((d-1))
    normx = np.ones((d-1))
    nrmsc = 1.0

    if verbose:
        print('Starting AMEn solve with:\n\tepsilon: %g\n\tsweeps: %d' % (eps, nswp))
        print()

    for swp in range(nswp):
        # right to left orthogonalization

        if verbose:
            print('Starting sweep %d %s...' %
                  (swp+1, "(last one) " if last else ""))
            tme_sweep = datetime.datetime.now()

        for k in range(d-1, 0, -1):

            # update the z part (ALS) update
            if not last:
                if swp > 0:
                    # shape rzp x N x rz
                    czA = _local_product(
                        Phiz[k+1], Phiz[k], A[k], x_cores[k], x_cores[k].shape, band_diagonal)
                    # shape is rzp x N x rz
                    czy = np.einsum('br,bnB,BR->rnR',Phiz_b[k], b[k], Phiz_b[k+1])
                    cz_new = czy*nrmsc - czA
                    _, _, vz = np.linalg.svd(np.reshape(cz_new, [cz_new.shape[0], -1]), full_matrices=False)
                    # truncate to kickrank
                    cz_new = vz[:min(kickrank, vz.shape[0]), :].T
                    if k < d-1:  # extend cz_new with random elements
                        cz_new = np.concatenate((cz_new, np.random.randn(cz_new.shape[0], kick2)), 1)
                else:
                    cz_new = np.reshape(z_cores[k], (rz[k], -1)).T

                qz, _ = np.linalg.qr(cz_new)
                rz[k] = qz.shape[1]
                z_cores[k] = np.reshape(qz.T, (rz[k], N[k], -1))

            # norm correction ?
            if swp > 0:
                nrmsc = nrmsc * normA[k-1] * normx[k-1] / normb[k-1]

            core = np.reshape(x_cores[k], [rx[k], N[k]*rx[k+1]]).T
            Qmat, Rmat = np.linalg.qr(core)

            core_prev = np.einsum('ijk,km->ijm', x_cores[k-1], Rmat.T)
            rx[k] = Qmat.shape[1]

            current_norm = np.linalg.norm(core_prev)
            if current_norm > 0:
                core_prev = core_prev / current_norm
            else:
                current_norm = 1.0
            normx[k-1] = normx[k-1]*current_norm

            x_cores[k] = np.reshape(Qmat.T, (rx[k], N[k], rx[k+1]))
            x_cores[k-1] = core_prev[:]

            # update phis (einsum)
            Phis[k] = _compute_phi_bck_A(Phis[k+1], x_cores[k], A[k], x_cores[k])
            Phis_b[k] = _compute_phi_bck_rhs(Phis_b[k+1], b[k], x_cores[k])

            # ... and norms
            norm = np.linalg.norm(Phis[k])
            norm = norm if norm > 0 else 1.0
            normA[k-1] = norm
            Phis[k] = Phis[k] / norm
            norm = np.linalg.norm(Phis_b[k])
            norm = norm if norm > 0 else 1.0
            normb[k-1] = norm
            Phis_b[k] = Phis_b[k]/norm

            # norm correction
            nrmsc = nrmsc * normb[k-1] / (normA[k-1] * normx[k-1])

            # compute phis_z
            if not last:
                Phiz[k] = _compute_phi_bck_A(Phiz[k+1], z_cores[k], A[k], x_cores[k]) / normA[k-1]
                Phiz_b[k] = _compute_phi_bck_rhs(Phiz_b[k+1], b[k], z_cores[k]) / normb[k-1]

        # start loop
        max_res = 0
        max_dx = 0

        for k in range(d):
            if verbose:
                print('\tCore', k)
            previous_solution = np.reshape(x_cores[k], [-1, 1])

            # assemble rhs
            rhs = np.einsum('br,bmB,BR->rmR',Phis_b[k], b[k] * nrmsc, Phis_b[k+1])
            rhs = np.reshape(rhs, [-1, 1])
            norm_rhs = np.linalg.norm(rhs)

            # residuals
            real_tol = (eps/np.sqrt(d))/damp

            # solve the local system
            # shape is Rp x N x N x r x r
            Bp = np.einsum('smnS,LSR->smnRL', A[k], Phis[k+1])
            B = np.einsum('lsr,smnRL->lmLrnR', Phis[k], Bp)
            B = np.reshape(B, [rx[k]*N[k]*rx[k+1], rx[k]*N[k]*rx[k+1]])

            solution_now = scip.linalg.solve(B, rhs)

            res_old = np.linalg.norm(B @ previous_solution-rhs)/norm_rhs
            res_new = np.linalg.norm(B @ solution_now-rhs)/norm_rhs

            # compute residual and step size
            dx = np.linalg.norm(solution_now-previous_solution) / np.linalg.norm(solution_now)
            if verbose:
                print('\t\tdx = %g, res_now = %g, res_old = %g' %
                      (dx, res_new, res_old))

            max_dx = max(dx, max_dx)
            max_res = max(max_res, res_old)

            solution_now = np.reshape(solution_now, (rx[k]*N[k], rx[k+1]))
            # truncation
            if k < d-1:
                u, s, v = np.linalg.svd(solution_now, full_matrices=False)
                if trunc_norm != 'fro':
                    # search for a rank such that offers small enough residuum
                    r = 0
                    for r in range(u.shape[1]-1, 0, -1):
                        # solution has the same size
                        solution = u[:, :r] @ np.diag(s[:r]) @ v[:r, :]
                        res = np.linalg.norm(np.reshape(_local_product(Phis[k+1],Phis[k],A[k],np.reshape(solution,[rx[k],N[k],rx[k+1]]),solution_now.shape),[-1,1]) - rhs)/norm_rhs

                        if res > max(real_tol*damp, res_new):
                            break
                    r = min(r+1, np.prod(s.shape))
            else:
                u, v = np.linalg.qr(solution_now)
                r = u.shape[1]
                s = np.ones(r,  dtype=dtype)

            u = u[:, :r]
            v = np.diag(s[:r]) @ v[:r, :]
            v = v.T

            if not last:
                czA = _local_product(
                    Phiz[k+1],
                    Phiz[k],
                    A[k],
                    np.reshape(u @ v.T, [rx[k], N[k], rx[k+1]]),
                    (rx[k], N[k], rx[k+1]),
                    band_diagonal
                )  # shape rzp x N x rz

                # shape is rzp x N x rz
                czy = np.einsum('br,bnB,BR->rnR',Phiz_b[k], b[k]*nrmsc, Phiz_b[k+1])
                cz_new = czy - czA
                uz, _, _ = np.linalg.svd(np.reshape(cz_new, (rz[k]*N[k], -1)), full_matrices=False)
                # truncate to kickrank
                cz_new = uz[:, :min(kickrank, uz.shape[1])]
                if k < d-1:  # extend cz_new with random elements
                    cz_new = np.concatenate((cz_new, np.random.randn(cz_new.shape[0], kick2)), 1)

                qz, _ = np.linalg.qr(cz_new)
                rz[k+1] = qz.shape[1]
                z_cores[k] = np.reshape(qz, (rz[k], N[k], rz[k+1]))

            if k < d-1:
                if not last:
                    left_res = _local_product(Phiz[k+1], Phis[k], A[k], np.reshape(
                        u @ v.T, [rx[k], N[k], rx[k+1]]), [rx[k], N[k], rx[k+1]], band_diagonal)
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
                Phis[k+1] = _compute_phi_fwd_A(Phis[k], x_cores[k], A[k], x_cores[k])
                Phis_b[k+1] = _compute_phi_fwd_rhs(Phis_b[k], b[k], x_cores[k])

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
                if not last:
                    Phiz[k+1] = _compute_phi_fwd_A(Phiz[k], z_cores[k],
                                                   A[k], x_cores[k]) / normA[k]
                    Phiz_b[k+1] = _compute_phi_fwd_rhs(
                        Phiz_b[k], b[k], z_cores[k]) / normb[k]
            else:
                x_cores[k] = np.reshape(u @ np.diag(s[:r]) @ v[:r, :].T, (rx[k], N[k], rx[k+1]))

        if verbose:
            print('Solution rank is', rx)
            print('Maxres ', max_res)
            tme_sweep = datetime.datetime.now()-tme_sweep
            print('Time ', tme_sweep)

        if last:
            break

        if max_res < eps:
            last = True

    if verbose:
        time_total = datetime.datetime.now() - time_total
        print()
        print('Finished after', swp+1, ' sweeps and ', time_total)
        print()

    normx = np.exp(np.sum(np.log(normx))/d)

    x_cores = [x_cores[k] * normx for k in range(d)]

    return x_cores, max_res


def _compute_phi_bck_A(Phi_now, core_left, core_A, core_right):
    Phi = np.einsum('LSR,lML,sMNS,rNR->lsr', Phi_now, core_left, core_A, core_right)
    return Phi


def _compute_phi_fwd_A(Phi_now, core_left, core_A, core_right):
    Phi_next = np.einsum('lsr,lML,sMNS,rNR->LSR', Phi_now, core_left, core_A, core_right)
    return Phi_next


def _compute_phi_bck_rhs(Phi_now, core_b, core):
    Phi = np.einsum('BR,bnB,rnR->br', Phi_now, core_b, core)
    return Phi


def _compute_phi_fwd_rhs(Phi_now, core_rhs, core):
    Phi_next = np.einsum('br,bnB,rnR->BR', Phi_now, core_rhs, core)
    return Phi_next


if __name__ == "__main__":
    np.random.seed(158)
    L = tt_rank_reduce(tt_random_binary([3, 4], shape=(4, 4)))
    initial_guess = tt_scale(5, tt_random_gaussian([4, 2], shape=(4,)))
    B = tt_rank_reduce(tt_matrix_vec_mul(L, initial_guess))
    res = tt_sub(tt_matrix_vec_mul(L, tt_one(len(L), shape=(4, ))), B)
    print("Start Error: ", tt_inner_prod(res, res))
    # L = tt_add(L, [(1e-6)*np.eye(4).reshape(1, 4, 4, 1) for _ in range(len(L))])
    t0 = time.time()
    solution = tt_amen(L, B, verbose=True)
    t1 = time.time()
    res = tt_sub(tt_matrix_vec_mul(L, solution), B)
    print(f"Time taken: {t1 - t0}s")
    print("Error: ", tt_inner_prod(res, res))
