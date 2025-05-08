import sys
import os
import time
import copy

import numpy as np

sys.path.append(os.getcwd() + '/../')

from src.tt_ops import *
from opt_einsum import contract as einsum



def _local_product(Phi_right, Phi_left, coreA, core):

    w = einsum('lsr,smnS,LSR,rnR->lmL', Phi_left, coreA, Phi_right, core)
    return w


def tt_amen(A, b, nswp=50, x0=None, eps=1e-10, rmax=1024, solver_limit=500, kickrank=2, verbose=False):

    dtype = A[0].dtype
    damp = 2

    if x0 == None:
        x = [np.ones((1, *c.shape[1:-1], 1), dtype=dtype) for c in b]
    else:
        x = x0

    x_cores = copy.copy(x)
    N = [c.shape[1] for c in x_cores]
    d = len(N)
    rx = [1] + tt_ranks(x) + [1]
    rmax = [1] + (d - 1) * [rmax] + [1]

    # z cores
    z_cores = tt_random_gaussian([kickrank] * (d - 1), shape=x_cores[0].shape[1:-1])
    z_cores = tt_rl_random_orthogonalise(z_cores, (d - 1) * [kickrank])
    rz = [1] + tt_ranks(z_cores) + [1]

    ZAX = [np.ones((1, 1, 1), dtype=dtype)] + [None] * (d - 1) + [np.ones((1, 1, 1), dtype=dtype)]  # size is rzk x Rk x rxk
    Zb = [np.ones((1, 1), dtype=dtype)] + [None] * (d - 1) + [np.ones((1, 1), dtype=dtype)]  # size is rzk x rzbk

    XAX = [np.ones((1, 1, 1), dtype=dtype)] + [None] * (d - 1) + [np.ones((1, 1, 1), dtype=dtype)]  # size is rk x Rk x rk
    Xb = [np.ones((1, 1), dtype=dtype)] + [None] * (d - 1) + [np.ones((1, 1), dtype=dtype)]  # size is rk x rbk

    last = False

    normA = np.ones((d - 1))
    normb = np.ones((d - 1))
    normx = np.ones((d - 1))
    nrmsc = 1.0

    if verbose:
        t0 = time.time()
        print('Starting AMEn solve with:\n\tEpsilon: %g\n\tMax num of sweeps: %d' % (eps, nswp))
        print(f"\tTT-Matrix rank: {tt_ranks(A)}")
        print(f"\tTT-bias rank: {tt_ranks(b)}")

    for swp in range(nswp):
        # right to left orthogonalization
        for k in range(d - 1, 0, -1):

            # update the z part (ALS) update
            if not last:
                if swp > 0:
                    # shape rzp x N x rz
                    czA = _local_product(ZAX[k + 1], ZAX[k], A[k], x_cores[k])
                    # shape is rzp x N x rz
                    czy = einsum('br,bnB,BR->rnR',
                                    Zb[k], b[k], Zb[k + 1])
                    cz_new = czy * nrmsc - czA
                    _, _, vz = scip.linalg.svd(np.reshape(cz_new, [cz_new.shape[0], -1]), full_matrices=False, check_finite=False)
                    # truncate to kickrank
                    cz_new = vz[:min(kickrank, vz.shape[0]), :].T
                else:
                    cz_new = np.reshape(z_cores[k], [rz[k], -1]).T
                qz, _ = np.linalg.qr(cz_new)
                rz[k] = qz.shape[1]
                z_cores[k] = np.reshape(qz.T, [rz[k], N[k], rz[k + 1]])

            # norm correction ?
            if swp > 0:
                nrmsc = nrmsc * normA[k - 1] * normx[k - 1] / normb[k - 1]

            core = np.reshape(x_cores[k], [rx[k], N[k] * rx[k + 1]]).T
            Qmat, Rmat = np.linalg.qr(core)

            core_prev = einsum('ijk,km->ijm', x_cores[k - 1], Rmat.T)
            rx[k] = Qmat.shape[1]

            current_norm = np.linalg.norm(core_prev)
            current_norm = current_norm if current_norm > 0 else 1.0
            core_prev = core_prev / current_norm

            normx[k - 1] = normx[k - 1] * current_norm

            x_cores[k] = np.reshape(Qmat.T, [rx[k], N[k], rx[k + 1]])
            x_cores[k - 1] = core_prev[:]

            # update phis (einsum)
            XAX[k] = _compute_phi_bck_A(
                XAX[k + 1], x_cores[k], A[k], x_cores[k])
            Xb[k] = _compute_phi_bck_rhs(
                Xb[k + 1], b[k], x_cores[k])

            # ... and norms
            norm = np.linalg.norm(XAX[k])
            norm = norm if norm > 0 else 1.0
            normA[k - 1] = norm
            XAX[k] = XAX[k] / norm
            norm = np.linalg.norm(Xb[k])
            norm = norm if norm > 0 else 1.0
            normb[k - 1] = norm
            Xb[k] = Xb[k] / norm

            # norm correction
            nrmsc = nrmsc * normb[k - 1] / (normA[k - 1] * normx[k - 1])

            # compute phis_z
            if not last:
                ZAX[k] = _compute_phi_bck_A(
                    ZAX[k + 1], z_cores[k], A[k], x_cores[k]) / normA[k - 1]
                Zb[k] = _compute_phi_bck_rhs(
                    Zb[k + 1], b[k], z_cores[k]) / normb[k - 1]

        # start loop
        max_res = 0

        for k in range(d):
            previous_solution = np.reshape(x_cores[k], [-1, 1])

            # assemble rhs
            rhs = einsum('br,bmB,BR->rmR',
                            Xb[k], b[k] * nrmsc, Xb[k + 1])
            rhs = np.reshape(rhs, [-1, 1])
            norm_rhs = np.linalg.norm(rhs)

            # residuals
            real_tol = (eps / np.sqrt(d)) / damp

            # solve the local system
            B = einsum('lsr,smnS,LSR->lmLrnR', XAX[k], A[k], XAX[k + 1])
            B = np.reshape(B, [rx[k] * N[k] * rx[k + 1], rx[k] * N[k] * rx[k + 1]])

            res_old = np.linalg.norm(B @ previous_solution - rhs) / norm_rhs

            # Avoid problems with ill-conditiong by using varaints of svd
            u, s, v = scip.linalg.svd(B, full_matrices=False, check_finite=False)
            s = s[s > real_tol]
            r = len(s)
            solution_now = v[:r].T @ np.diag(np.divide(1, s)) @ u[:, :r].T @ rhs

            res_new = np.linalg.norm(B @ solution_now - rhs) / norm_rhs

            # residual damp check
            if res_old < damp*res_new and res_new > real_tol:
                if verbose:
                    print('WARNING: residual increases. res_old %g, res_new %g, real_tol %g' % (
                        res_old, res_new, real_tol))  # warning (from tt toolbox)

            max_res = max(max_res, res_old)

            solution_now = np.reshape(solution_now, [rx[k] * N[k], rx[k + 1]])
            # truncation
            if k < d - 1:
                u, s, v = scip.linalg.svd(solution_now, full_matrices=False, check_finite=False)
                r = 0
                for r in range(u.shape[1] - 1, 0, -1):
                    # solution has the same size
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
                czA = _local_product(ZAX[k + 1], ZAX[k], A[k],
                                     np.reshape(u @ v.T, [rx[k], N[k], rx[k + 1]]))  # shape rzp x N x rz

                # shape is rzp x N x rz
                czy = einsum('br,bnB,BR->rnR', Zb[k], b[k] * nrmsc, Zb[k + 1])
                cz_new = czy - czA
                uz, _, _ = scip.linalg.svd(np.reshape(cz_new, [rz[k] * N[k], rz[k + 1]]), full_matrices=False, check_finite=False)
                # truncate to kickrank
                cz_new = uz[:, :min(kickrank, uz.shape[1])]

                qz, _ = np.linalg.qr(cz_new)
                rz[k + 1] = qz.shape[1]
                z_cores[k] = np.reshape(qz, [rz[k], N[k], rz[k + 1]])

            if k < d - 1:
                if not last:
                    left_res = _local_product(ZAX[k + 1], XAX[k], A[k], np.reshape(u @ v.T, [rx[k], N[k], rx[k + 1]]))
                    left_b = einsum('br,bmB,BR->rmR', Xb[k], b[k] * nrmsc, Zb[k + 1])
                    uk = left_b - left_res  # rx_k x N_k x rz_k+1
                    u, Rmat = np.linalg.qr(np.concatenate((u, np.reshape(uk, [rx[k]*N[k], -1])), 1))
                    r_add = uk.shape[2]
                    v = np.concatenate((v, np.zeros([rx[k + 1], r_add], dtype=dtype)), 1)
                    v = v @ Rmat.T

                r = u.shape[1]
                v = einsum('ji,jkl->ikl', v, x_cores[k + 1])
                # remove norm correction
                nrmsc = nrmsc * normA[k] * normx[k] / normb[k]

                norm_now = np.linalg.norm(v)
                norm_now = norm_now if norm_now > 0 else 1.0
                v = v / norm_now
                normx[k] = normx[k] * norm_now

                x_cores[k] = np.reshape(u, [rx[k], N[k], r])
                x_cores[k + 1] = np.reshape(v, (r, N[k + 1], rx[k + 2]))
                rx[k + 1] = r
                # next phis with norm correction
                XAX[k + 1] = _compute_phi_fwd_A(XAX[k], x_cores[k], A[k], x_cores[k])
                Xb[k + 1] = _compute_phi_fwd_rhs(Xb[k], b[k], x_cores[k])

                # ... and norms
                norm = np.linalg.norm(XAX[k + 1])
                norm = norm if norm > 0 else 1.0
                normA[k] = norm
                XAX[k + 1] = XAX[k + 1] / norm
                norm = np.linalg.norm(Xb[k + 1])
                norm = norm if norm > 0 else 1.0
                normb[k] = norm
                Xb[k + 1] = Xb[k + 1] / norm
                # norm correction
                nrmsc = nrmsc * normb[k] / (normA[k] * normx[k])

                # next phiz
                if not last:
                    ZAX[k + 1] = _compute_phi_fwd_A(ZAX[k], z_cores[k],
                                                     A[k], x_cores[k]) / normA[k]
                    Zb[k + 1] = _compute_phi_fwd_rhs(
                        Zb[k], b[k], z_cores[k]) / normb[k]
            else:
                x_cores[k] = np.reshape(
                    u @ np.diag(s[:r]) @ v[:r, :].T, [rx[k], N[k], rx[k + 1]])

        if last:
            break

        if max_res < eps:
            last = True

    if verbose:
        print("\t---Results---")
        print('\tSolution rank is', rx[1:-1])
        print('\tResidual ', max_res)
        print('\tNumber of sweeps', swp+1)
        print('\tTime: ', time.time() - t0)
        print('\tTime per sweep: ', (time.time() - t0) / (swp+1))

    normx = np.exp(np.sum(np.log(normx)) / d)

    for k in range(d):
        x_cores[k] = x_cores[k] * normx

    return x_cores, max_res

def _compute_phi_bck_A(Phi_now, core_left, core_A, core_right):
    return cached_einsum('LSR,lML,sMNS,rNR->lsr', Phi_now, core_left, core_A, core_right)


def _compute_phi_fwd_A(Phi_now, core_left, core_A, core_right):
    return cached_einsum('lsr,lML,sMNS,rNR->LSR',Phi_now, core_left, core_A, core_right)


def _compute_phi_bck_rhs(Phi_now, core_b, core):
    return cached_einsum('BR,bnB,rnR->br', Phi_now, core_b, core)


def _compute_phi_fwd_rhs(Phi_now, core_rhs, core):
    return cached_einsum('br,bnB,rnR->BR', Phi_now, core_rhs, core)


def _bck_sweep(
        local_solver,
        x_cores,
        normx,
        XAX,
        block_A,
        normA,
        Xb,
        block_b,
        normb,
        nrmsc,
        rx,
        N,
        block_size,
        real_tol,
        d,
        swp,
        size_limit,
        eps,
        r_max
):
    local_res = np.inf if swp == 0 else 0
    local_dx = np.inf if swp == 0 else 0
    for k in range(d - 1, -1, -1):
        if swp > 0:
            previous_solution = x_cores[k]
            solution_now, block_res_old, block_res_new, rhs, norm_rhs = local_solver(XAX[k], block_A[k], XAX[k + 1],
                                                                                     Xb[k], block_b[k], Xb[k + 1],
                                                                                     previous_solution, nrmsc,
                                                                                     size_limit, eps)

            local_res = max(local_res, block_res_old)
            dx = np.linalg.norm(solution_now - previous_solution) / np.linalg.norm(solution_now)
            local_dx = max(dx, local_dx)

            solution_now = np.reshape(solution_now, (rx[k] * block_size, N[k] * rx[k + 1])).T

        else:
            solution_now = np.reshape(x_cores[k], (rx[k] * block_size, N[k] * rx[k + 1])).T

        if k > 0:
            if min(rx[k] * block_size, N[k] * rx[k + 1]) > 2*r_max:
                u, s, v = scip.sparse.linalg.svds(solution_now, k=r_max, tol=eps, which="LM")
                idx = np.argsort(s)[::-1]  # descending order
                s = s[idx]
                u = u[:, idx]
                v = v[idx, :]
            else:
                u, s, v = scip.linalg.svd(solution_now, full_matrices=False, check_finite=False, overwrite_a=True)
            v = s.reshape(-1, 1) * v
            if swp > 0:
                r_start = min(prune_singular_vals(s, real_tol), r_max)
                solution_now = np.reshape((u[:, :r_start] @ v[:r_start]).T, (rx[k], block_size, N[k], rx[k + 1]))
                res = _block_local_product(XAX[k], block_A[k], XAX[k + 1], solution_now) - rhs
                r = r_start
                for r in range(r_start - 1, 0, -1):
                    res -= _block_local_product(XAX[k], block_A[k], XAX[k + 1],
                                                np.reshape((u[:, None, r] @ v[None, r, :]).T,
                                                           (rx[k], block_size, N[k], rx[k + 1])))
                    if np.linalg.norm(res) / norm_rhs > max(2 * real_tol, block_res_new):
                        break
                r += 1
                nrmsc *= (normA[k - 1] * normx[k - 1]) / normb[k - 1]
            else:
                r = min(prune_singular_vals(s, real_tol), r_max)
            x_cores[k] = np.reshape(u[:, :r].T, (r, N[k], rx[k + 1]))
            x_cores[k - 1] = einsum('rdc,cbR->rbdR', x_cores[k - 1], v[:r].T.reshape(rx[k], block_size, r), optimize="greedy")
            norm_now = np.linalg.norm(x_cores[k - 1])
            normx[k - 1] *= norm_now
            x_cores[k - 1] /= norm_now
            rx[k] = r

            XAX[k] = {(i, j): _compute_phi_bck_A(XAX[k + 1][(i, j)], x_cores[k], block_A[k][(i, j)], x_cores[k]) for (i, j)
                      in block_A[k]}
            normA[k - 1] = max(np.sqrt(sum(np.sum(v ** 2) for v in XAX[k].values())), eps)
            XAX[k] = {(i, j): XAX[k][(i, j)] / normA[k - 1] for (i, j) in block_A[k]}

            Xb[k] = {i: _compute_phi_bck_rhs(Xb[k + 1][i], block_b[k][i], x_cores[k]) for i in block_b[k]}
            normb[k - 1] = max(np.sqrt(sum(np.sum(v ** 2) for v in Xb[k].values())), eps)
            Xb[k] = {i: Xb[k][i] / normb[k - 1] for i in block_b[k]}
            nrmsc *= normb[k - 1] / (normA[k - 1] * normx[k - 1])
        else:
            x_cores[k] = np.reshape(solution_now.T, (rx[k], block_size, N[k], rx[k + 1]))

    return x_cores, normx, XAX, normA, Xb, normb, nrmsc, rx, local_res, local_dx


def _fwd_sweep(
        local_solver,
        x_cores,
        normx,
        XAX,
        block_A,
        normA,
        Xb,
        block_b,
        normb,
        nrmsc,
        rx,
        N,
        block_size,
        real_tol,
        d,
        swp,
        size_limit,
        eps,
        r_max
):
    local_res = np.inf if swp == 0 else 0
    local_dx = np.inf if swp == 0 else 0
    for k in range(d):
        if swp > 0:
            previous_solution = x_cores[k]
            solution_now, block_res_old, block_res_new, rhs, norm_rhs = local_solver(
                XAX[k], block_A[k], XAX[k + 1], Xb[k],
                block_b[k], Xb[k + 1],
                previous_solution, nrmsc, size_limit,
                eps
            )

            local_res = max(local_res, block_res_old)
            dx = np.linalg.norm(solution_now - previous_solution) / np.linalg.norm(solution_now)
            local_dx = max(dx, local_dx)

            solution_now = np.transpose(solution_now, (0, 2, 1, 3))
            solution_now = np.reshape(solution_now, (rx[k] * N[k], block_size * rx[k + 1]))
        else:
            solution_now = np.reshape(x_cores[k], (rx[k] * N[k],  block_size * rx[k + 1]))

        if k < d - 1:
            if min(rx[k] * N[k],  block_size * rx[k + 1]) > 2*r_max:
                u, s, v = scip.sparse.linalg.svds(solution_now, k=r_max, tol=eps, which="LM")
                idx = np.argsort(s)[::-1]  # descending order
                s = s[idx]
                u = u[:, idx]
                v = v[idx, :]
            else:
                u, s, v = scip.linalg.svd(solution_now, full_matrices=False, check_finite=False, overwrite_a=True)
            v = s.reshape(-1, 1) * v
            u = u.reshape(rx[k], N[k], -1)
            v = v.reshape(-1, block_size, rx[k + 1])
            if swp > 0:
                r_start = min(prune_singular_vals(s, real_tol), r_max)
                solution_now = einsum("rbR, Rdk -> rbdk", u[:, :, :r_start], v[:r_start], optimize="greedy")
                res = _block_local_product(XAX[k], block_A[k], XAX[k + 1], np.transpose(solution_now, (0, 2, 1, 3))) - rhs
                r = r_start
                for r in range(r_start - 1, 0, -1):
                    res -= _block_local_product(XAX[k], block_A[k], XAX[k + 1], np.transpose(
                        einsum("rbR, Rdk -> rbdk", u[:, :, None, r], v[None, r], optimize="greedy"), (0, 2, 1, 3)))
                    if np.linalg.norm(res) / norm_rhs > max(2 * real_tol, block_res_new):
                        break
                r += 1
                nrmsc *= normA[k] * normx[k] / normb[k]
            else:
                r = min(prune_singular_vals(s, real_tol), r_max)

            v = einsum("rbR, Rdk -> rbdk", v[:r], x_cores[k + 1], optimize="greedy")
            norm_now = np.linalg.norm(v)
            normx[k] *= norm_now
            x_cores[k] = u[:, :, :r]
            x_cores[k + 1] = (v / norm_now).reshape(r, block_size, N[k + 1], rx[k + 2])
            rx[k + 1] = r

            XAX[k + 1] = {(i, j): _compute_phi_fwd_A(XAX[k][(i, j)], x_cores[k], block_A[k][(i, j)], x_cores[k]) for
                          (i, j) in block_A[k]}
            normA[k] = max(np.sqrt(sum(np.sum(v ** 2) for v in XAX[k + 1].values())), eps)
            XAX[k + 1] = {(i, j): XAX[k + 1][(i, j)] / normA[k] for (i, j) in block_A[k]}

            Xb[k + 1] = {i: _compute_phi_fwd_rhs(Xb[k][i], block_b[k][i], x_cores[k]) for i in block_b[k]}
            normb[k] = max(np.sqrt(sum(np.sum(v ** 2) for v in Xb[k + 1].values())), eps)
            Xb[k + 1] = {i: Xb[k + 1][i] / normb[k] for i in block_b[k]}

            nrmsc *= normb[k] / (normA[k] * normx[k])

        else:
            x_cores[k] = np.reshape(solution_now, (rx[k], N[k], block_size, rx[k + 1])).transpose(0, 2, 1, 3)

    return x_cores, normx, XAX, normA, Xb, normb, nrmsc, rx, local_res, local_dx


def tt_block_mals(block_A, block_b, tol, eps=1e-10, nswp=22, x0=None, local_solver=None, verbose=False):

    block_size = np.max(list(k[0] for k in block_A.keys())) + 1
    model_entry = next(iter(block_b.values()))
    x_shape = model_entry[0].shape[1:-1]

    if local_solver is None:
        local_solver = _default_local_solver

    direction = 1
    if x0 == None:
        # TODO: Temp decreased block_size
        x_cores = tt_normalise([np.random.randn(1, *c.shape[1:-1], 1) for c in model_entry[:-1]]) + [np.random.randn(1, block_size, *x_shape, 1)]
    else:
        if len(x0[0].shape) > len(x0[-1].shape):
            direction *= -1
        x_cores = x0

    if verbose:
        t0 = time.time()

    N = [c.shape[-2] for c in x_cores]
    d = len(N)
    rx = np.array([1] + tt_ranks(x_cores) + [1])


    XAX =  [{key: np.ones((1, 1, 1)) for key in block_A}] + [{key: None for key in block_A} for _ in range(d-1)] + [{key: np.ones((1, 1, 1)) for key in block_A}]  # size is rk x Rk x rk
    Xb = [{key: np.ones((1, 1)) for key in block_b}] + [{key: None for key in block_b} for _ in range(d-1)] + [{key: np.ones((1, 1)) for key in block_b}]   # size is rk x rbk
    block_A = [{(i, j): block_A[(i, j)][k] for (i,  j) in block_A} for k in range(d)]
    block_b = [{i: block_b[i][k] for i in block_b} for k in range(d)]

    normA = np.ones(d - 1) # norm of each row in the block matrix
    normb = np.ones(d - 1) # norm of each row of the rhs
    nrmsc = 1.0
    normx = np.ones((d - 1))
    real_tol = (tol / np.sqrt(d))
    r_max_final = block_size*int(np.sqrt(d)*d) + block_size
    size_limit = 0.1*r_max_final*N[0]*r_max_final
    r_max_part = max(int(r_max_final // (nswp-1)), np.max(tt_ranks(x_cores)))

    for swp in range(nswp):
        r_max = min((swp+1)*r_max_part, r_max_final)
        if direction > 0:
            x_cores, normx, XAX, normA, Xb, normb, nrmsc, rx, local_res, local_dx = _bck_sweep(
                local_solver,
                x_cores,
                normx,
                XAX,
                block_A,
                normA,
                Xb,
                block_b,
                normb,
                nrmsc,
                rx,
                N,
                block_size,
                real_tol,
                d,
                swp,
                size_limit,
                eps,
                r_max
            )
        else:
            x_cores, normx, XAX, normA, Xb, normb, nrmsc, rx, local_res, local_dx = _fwd_sweep(
                local_solver,
                x_cores,
                normx,
                XAX,
                block_A,
                normA,
                Xb,
                block_b,
                normb,
                nrmsc,
                rx,
                N,
                block_size,
                real_tol,
                d,
                swp,
                size_limit,
                eps,
                r_max
            )

        if verbose:
            print('Starting Sweep:\n\tMax num of sweeps: %d' % swp)
            print(f'\tDirection {direction}')
            print(f'\tResidual {local_res}')
            print(f"\tTT-sol rank: {tt_ranks(x_cores)}", flush=True)
        if local_res < tol or local_dx < eps:
            break

        direction *= -1


    if verbose:
        print("\n\t---Results---")
        print('\tSolution rank is', rx[1:-1])
        print('\tResidual ', local_res)
        print('\tNumber of sweeps', swp+1)
        print('\tTime: ', time.time() - t0)
        print('\tTime per sweep: ', (time.time() - t0) / (swp+1), flush=True)

    normx = np.exp(np.sum(np.log(normx)) / d)

    return [normx * core for core in x_cores], np.mean(local_res)


def tt_divide(vec_tt_1, vec_tt_2, degenerate=False, eps=1e-10):
    b = vec_tt_1
    A = tt_diag(vec_tt_2)
    if degenerate:
        A = tt_add(A, tt_scale(0.5*eps, tt_identity(len(A))))
    A = tt_rank_reduce(A, eps)
    sol, _ = tt_amen(A, b)
    return sol


def _block_local_product(XAX_k, block_A_k, XAX_kp1, x_core):
    result = np.zeros_like(x_core)
    for (i, j) in block_A_k:
        result[:, i] += cached_einsum('lsr,smnS,LSR,rnR->lmL', XAX_k[(i, j)], block_A_k[(i, j)], XAX_kp1[(i, j)], x_core[:, j])
    return result

def _default_local_solver(XAX_k, block_A_k, XAX_k1, Xb_k, block_b_k, Xb_k1, previous_solution, nrmsc, size_limit, rtol):
    x_shape = previous_solution.shape
    block_size = x_shape[1]
    m = x_shape[0]*x_shape[2]*x_shape[3]
    rhs = np.zeros_like(previous_solution)
    x_shape = (x_shape[1], x_shape[0], x_shape[2], x_shape[3])
    for i in block_b_k:
        rhs[:, i] = cached_einsum('br,bmB,BR->rmR', Xb_k[i], nrmsc * block_b_k[i], Xb_k1[i])
    norm_rhs = np.linalg.norm(rhs)
    if m <= size_limit:
        B = np.zeros((block_size * m, block_size * m))
        for (i, j) in block_A_k:
            local_B = cached_einsum('lsr,smnS,LSR->lmLrnR', XAX_k[(i, j)], block_A_k[(i, j)], XAX_k1[(i, j)])
            B[m * i:m * (i + 1), m * j:m * (j + 1)] = local_B.reshape(m, m)

        solution_now = np.transpose(scip.linalg.solve(B, np.transpose(rhs, (1, 0, 2, 3)).reshape(-1, 1) - B @ np.transpose(previous_solution, (1, 0, 2, 3)).reshape(-1, 1), check_finite=False).reshape(*x_shape), (1, 0, 2, 3))
    else:
        def mat_vec(x_vec):
            return np.transpose(_block_local_product(
                XAX_k, block_A_k, XAX_k1,
                np.transpose(x_vec.reshape(*x_shape), (1, 0, 2, 3))
            ), (1, 0, 2, 3)).reshape(-1, 1)

        linear_op = scip.sparse.linalg.LinearOperator((block_size * m, block_size * m), matvec=mat_vec)
        solution_now, info = scip.sparse.linalg.bicgstab(linear_op, np.transpose(
            rhs - _block_local_product(XAX_k, block_A_k, XAX_k1, previous_solution), (1, 0, 2, 3)).reshape(-1, 1), rtol=rtol)
        solution_now = np.transpose(solution_now.reshape(*x_shape), (1, 0, 2, 3))

    solution_now += previous_solution
    block_res_new = np.linalg.norm(
        _block_local_product(XAX_k, block_A_k, XAX_k1, solution_now) - rhs) / norm_rhs
    block_res_old = np.linalg.norm(
        _block_local_product(XAX_k, block_A_k, XAX_k1, previous_solution) - rhs) / norm_rhs

    if block_res_old < block_res_new:
        solution_now = previous_solution

    return solution_now, block_res_old, min(block_res_old, block_res_new), rhs, norm_rhs





