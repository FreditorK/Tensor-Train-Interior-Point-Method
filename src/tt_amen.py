import sys
import os

import numpy as np

sys.path.append(os.getcwd() + '/../')

from src.tt_ops import *
from src.tt_ops import _tt_lr_random_orthogonalise
import time
import sklearn
import copy
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
            B_size = B.shape[0]
            is_large = B_size > solver_limit
            sparsity_ratio = np.sum(np.abs(B) > real_tol) / B.size
            if is_large and sparsity_ratio < 0.25:
                num_sing_values = int(np.floor((2*sparsity_ratio * B_size)))
                u, s, v = sklearn.utils.extmath.randomized_svd(B, n_components=num_sing_values, random_state=0)
            else:
                u, s, v = scip.linalg.svd(B, full_matrices=False, check_finite=False)
            s = s[s > real_tol]
            r = len(s)
            solution_now = v[:r].T @ np.diag(np.divide(1, s)) @ u[:, :r].T @ rhs

            res_new = np.linalg.norm(B @ solution_now - rhs) / norm_rhs

            # residual damp check
            if res_old / res_new < damp and res_new > real_tol:
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
                uz, _, _ = scip.linalg.svd(np.reshape(cz_new, [rz[k] * N[k], rz[k + 1]]), full_matrices=False,
                                            check_finite=False)
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
    Phi = einsum('LSR,lML,sMNS,rNR->lsr', Phi_now,
                      core_left, core_A, core_right)
    return Phi


def _compute_phi_fwd_A(Phi_now, core_left, core_A, core_right):
    Phi_next = einsum('lsr,lML,sMNS,rNR->LSR',
                           Phi_now, core_left, core_A, core_right)
    return Phi_next


def _compute_phi_bck_rhs(Phi_now, core_b, core):
    Phi = einsum('BR,bnB,rnR->br', Phi_now, core_b, core)
    return Phi


def _compute_phi_fwd_rhs(Phi_now, core_rhs, core):
    Phi_next = einsum('br,bnB,rnR->BR', Phi_now, core_rhs, core)
    return Phi_next


def tt_inv_precond(matrix_tt, target_ranks, tol=1e-10, max_iter=100, verbose=False):
    if verbose:
        t0 = time.time()
    if np.all(np.array(tt_ranks(matrix_tt))== 1):
        pinv_tt = [np.linalg.pinv(np.squeeze(m)) for m in matrix_tt]
        return [m.reshape(1, *m.shape, 1) for m in pinv_tt], 0
    norm = np.sqrt(tt_inner_prod(matrix_tt, matrix_tt))
    inv_tt = tt_scale(np.divide(1, norm), tt_transpose(matrix_tt))
    tt_gaussian = tt_random_gaussian(target_ranks, shape=matrix_tt[0].shape[1:-1])
    identity = [np.eye(c.shape[1]).reshape(1, *c.shape[1:-1], 1) for c in matrix_tt]
    matrix_tt_t = tt_transpose(matrix_tt)
    prev_AG_ip = np.inf
    alpha = 0
    for _ in range(max_iter):
        R = tt_sub(identity, tt_mat_mat_mul(matrix_tt, inv_tt))
        G = tt_scale(-2, tt_mat_mat_mul(matrix_tt_t, R))
        AG = tt_mat_mat_mul(matrix_tt, G)
        AG_ip = tt_inner_prod(AG, AG)
        alpha = 0.5*alpha + 0.5*tt_inner_prod(R, AG)/AG_ip
        inv_tt = tt_add(inv_tt, tt_scale(alpha, G))
        inv_tt = _tt_lr_random_orthogonalise(inv_tt, tt_gaussian)
        if np.abs(AG_ip - prev_AG_ip) < tol:
            break
        prev_AG_ip = AG_ip

    if verbose:
        print(f"Time: {time.time()-t0}s")
    return inv_tt


def tt_block_amen(block_A, block_b, nswp=22, x0=None, eps=1e-10, rmax=1024, kickrank=2, amen=False, verbose=False):

    damp = 2
    block_size = np.max(list(block_b.keys())) + 1
    model_entry = next(iter(block_b.values()))
    x_shape = model_entry[0].shape[1:-1]

    if x0 == None:
        x = tt_normalise([np.random.randn(1, *c.shape[1:-1], 1) for c in model_entry[:-1]] + [np.random.randn(1, block_size, *x_shape, 1)])
    else:
        x = x0

    x_cores = copy.copy(x)
    N = [c.shape[-2] for c in x_cores]
    d = len(N)
    rx = [1] + tt_ranks(x) + [1]
    rmax = [1] + (d - 1) * [rmax] + [1]

    XAX = {key: [np.ones((1, 1, 1))] + [None] * (d - 1) + [np.ones((1, 1, 1))] for key in block_A} # size is rk x Rk x rk
    Xb = {key: [np.ones((1, 1))] + [None] * (d - 1) + [np.ones((1, 1))] for key in block_b}  # size is rk x rbk
    if amen:
        # z cores
        z_cores = tt_normalise(
            [np.random.randn(1, block_size, *x_shape, kickrank)]
            + [np.random.randn(kickrank, *c.shape[1:-1], kickrank) for c in model_entry[1:-1]]
            + [np.random.randn(kickrank, *x_shape, 1)]
        )
        z_cores = tt_rl_orthogonalise(z_cores)
        rz = [1] + tt_ranks(z_cores) + [1]
        ZAX = copy.deepcopy(XAX) # size is rzk x Rk x rxk
        Zb = copy.deepcopy(Xb) # size is rzk x rzbk

    last = False

    if verbose:
        t0 = time.time()
        print('Starting AMEn solve with:\n\tEpsilon: %g\n\tMax num of sweeps: %d' % (eps, nswp))
        A_ranks = {key: tt_ranks(block) for key, block in block_A.items()}
        print(f"\tTT-Matrix rank: {A_ranks}")
        b_ranks = {key: tt_ranks(block) for key, block in block_b.items()}
        print(f"\tTT-bias rank: {b_ranks} \n")

    normA = np.ones((block_size, d - 1)) # norm of each row in the block matrix
    normb = np.ones((block_size, d - 1)) # norm of each row of the rhs
    nrmsc = np.ones(block_size)
    normx = np.ones((d - 1))
    real_tol = (eps / np.sqrt(d)) / damp

    for swp in range(nswp):
        for k in range(d - 1, 0, -1):
            if swp > 0:
                nrmsc *= (normA[:, k - 1] * normx[k - 1]) / normb[:, k - 1]
            # update the z part (ALS) update
            if not last and amen:
                if swp > 0:
                    czA = np.zeros((rz[k], block_size, N[k], rz[k + 1]))
                    for (i, j) in block_A:
                        czA[:, i] += _local_product(ZAX[(i, j)][k + 1], ZAX[(i, j)][k], block_A[(i, j)][k], x_cores[k][:, j])
                    # shape is rzp x N x rz
                    czy = np.zeros((rz[k], block_size, N[k], rz[k + 1]))
                    for i in block_b:
                        czy[:, i] = einsum('br,bnB,BR->rnR', Zb[i][k], nrmsc[i] * block_b[i][k], Zb[i][k + 1])
                    cz_new = czy - czA
                    # push block to from right to left
                    uz, sz, vz = scip.linalg.svd(np.reshape(cz_new, (rz[k]*block_size, N[k]*rz[k+1])), full_matrices=False, check_finite=False)
                    # truncate to kickrank
                    trunc_r = min(kickrank, vz.shape[0])
                    cz_new = np.reshape(vz[:trunc_r, :], [trunc_r, N[k] * rz[k + 1]]).T
                    z_cores[k - 1] = einsum('rdc,cbR->rbdR', z_cores[k - 1], (uz[:, :trunc_r] @ np.diag(sz[:trunc_r])).T.reshape(rz[k], block_size, trunc_r))
                else:
                    cz_new = np.reshape(z_cores[k], (rz[k], N[k]*rz[k+1])).T
                    trunc_r = rz[k]

                # right to left orthogonalisation of z_cores
                Qz, _ = np.linalg.qr(cz_new)
                z_cores[k] = np.reshape(Qz.T, (trunc_r, N[k], rz[k+1]))
                rz[k] = trunc_r

            # right to left orthogonalisation of x_cores
            core = np.reshape(x_cores[k], [rx[k]*block_size, N[k]*rx[k + 1]]).T
            Qmat, Rmat = np.linalg.qr(core)
            shifted_r = Qmat.shape[1]

            x_cores[k] = np.reshape(Qmat.T, [shifted_r, N[k], rx[k + 1]])
            x_cores[k - 1] = einsum('rdc,cbR->rbdR', x_cores[k - 1], Rmat.T.reshape(rx[k], block_size, shifted_r))
            norm_now = np.linalg.norm(x_cores[k-1])
            x_cores[k - 1] /= norm_now
            normx[k-1] *= norm_now
            rx[k] = shifted_r

            row_A = np.zeros(block_size)
            for (i, j) in block_A:
                # update phis (einsum)
                XAX_ij_k = _compute_phi_bck_A(XAX[(i, j)][k + 1], x_cores[k], block_A[(i, j)][k], x_cores[k]) # rx[k] x rA[k] x rx[k]
                row_A[i] += np.linalg.norm(XAX_ij_k)**2
                XAX[(i, j)][k] = XAX_ij_k
                if not last and amen:
                    ZAX[(i, j)][k] = _compute_phi_bck_A(ZAX[(i, j)][k + 1], z_cores[k], block_A[(i, j)][k], x_cores[k]) # rz[k] x rA[k] x rx[k]
            row_A += (row_A < real_tol)
            normA[:, k - 1] = np.sqrt(row_A)
            for (i, j) in block_A:
                XAX[(i, j)][k] /= normA[i, k - 1]
                if not last and amen:
                    ZAX[(i, j)][k] /= normA[i, k - 1]

            for i in block_b:
                Xb_i_k = _compute_phi_bck_rhs(Xb[i][k + 1], block_b[i][k], x_cores[k]) # rb[k] x rx[k]
                norm = np.linalg.norm(Xb_i_k)
                norm = norm if norm > real_tol else 1.0
                normb[i, k - 1] = norm
                Xb[i][k] = Xb_i_k / norm
                if not last and amen:
                    Zb[i][k] = _compute_phi_bck_rhs(Zb[i][k + 1], block_b[i][k], z_cores[k]) / norm # rb[k] x rz[k]
            nrmsc *= normb[:, k - 1] / (normA[:, k - 1] * normx[k - 1])

        # start loop
        max_res = 0
        max_dx = 0

        for k in range(d):

            # bring block dimension to front
            previous_solution = np.reshape(np.transpose(x_cores[k], (1, 0, 2, 3)), (-1, 1))
            m = rx[k] * N[k] * rx[k + 1]

            # assemble rhs
            rhs = np.zeros((block_size*m, 1))
            for i in block_b:
                local_rhs = einsum('br,bmB,BR->rmR', Xb[i][k], nrmsc[i] * block_b[i][k], Xb[i][k + 1])
                rhs[m*i: m*(i+1)] = np.reshape(local_rhs, (-1, 1))
            norm_rhs = np.linalg.norm(rhs)

            # residuals
            norm_rhs = norm_rhs if norm_rhs > real_tol else 1.0

            # assemble lhs
            B = np.zeros((block_size*m, block_size*m))
            for (i, j) in block_A:
                local_B = einsum('lsr,smnS,LSR->lmLrnR', XAX[(i, j)][k], block_A[(i, j)][k], XAX[(i, j)][k + 1])
                B[m*i:m*(i+1), m*j:m*(j+1)] = local_B.reshape(m, m)

            # Solve block system
            #solution_now = svd_solve_local_system(B, rhs, eps)
            solution_now =  ipm_solve_local_system(B, rhs, m,  block_size, eps)

            block_res_new = np.linalg.norm(B @ solution_now - rhs) / norm_rhs
            block_res_old = np.linalg.norm(B @ previous_solution - rhs) / norm_rhs

            # residual damp check
            if block_res_old / block_res_new < damp and block_res_new > real_tol:
                if verbose:
                    print(f"\r\tWARNING: residual increases. {block_res_old:10f}, {block_res_new:10f}", end='', flush=True)  # warning (from tt toolbox)

            max_res = max(max_res, block_res_old)
            #print(k, block_res_old, block_res_new, block_res_old / block_res_new)

            dx = np.linalg.norm(solution_now - previous_solution) / np.linalg.norm(solution_now)
            max_dx = max(max_dx, dx)

            solution_now = np.reshape(solution_now, (block_size, rx[k], N[k], rx[k + 1]))
            solution_now = np.transpose(solution_now, (1, 2, 0, 3))
            solution_now = np.reshape(solution_now, [rx[k] * N[k] * block_size, rx[k + 1]])

            # solution truncation
            if k < d - 1:
                u, s, v = scip.linalg.svd(solution_now, full_matrices=False, check_finite=False)
                v = np.diag(s) @ v
                r = 0
                for r in range(u.shape[1] - 1, 0, -1):
                    solution = np.reshape(u[:, :r] @ v[:r, :], (rx[k], N[k], block_size, rx[k + 1]))
                    res = np.linalg.norm(B @ np.reshape(np.transpose(solution, (2, 0, 1, 3)), (-1, 1)) - rhs) / norm_rhs
                    if res > max(real_tol * damp, block_res_new):
                        break
                r += 1
                r = min([r, np.size(s), rmax[k + 1]])
                u = u[:, :r]
                v = v[:r, :]
            else:
                u, v = np.linalg.qr(solution_now)

            if not last and amen:
                # Computing local residuals
                czA = np.zeros((rz[k], block_size, N[k], rz[k + 1]))
                local_core = np.reshape(u @ v, [rx[k], N[k], block_size, rx[k + 1]])
                for (i, j) in block_A:
                    czA[:, i] += _local_product(ZAX[(i, j)][k + 1], ZAX[(i, j)][k], block_A[(i, j)][k], local_core[:, :, j])  # shape rzp x N x rz
                czy = np.zeros((rz[k], block_size, N[k], rz[k + 1]))
                for i in block_b:
                    czy[:, i] = einsum('br,bnB,BR->rnR', Zb[i][k], nrmsc[i] * block_b[i][k], Zb[i][k + 1])
                cz_new = czy - czA
                if k < d - 1:
                    # push block to right so it does not appear in uz
                    cz_new = np.transpose(cz_new, (0, 2, 1, 3))
                    uz, sz, vz = scip.linalg.svd(np.reshape(cz_new, (rz[k] * N[k], block_size * rz[k + 1])), full_matrices=False, check_finite=False)
                    # truncate to kickrank
                    trunc_r = min(kickrank, uz.shape[1])
                    cz_new = uz[:, :trunc_r]
                    cz_new = np.reshape(cz_new, (rz[k] * N[k], trunc_r))

                    # lr orthogonalise and shift block to right
                    Qz, Rz = np.linalg.qr(cz_new)
                    z_cores[k] = np.reshape(Qz, (rz[k], N[k], trunc_r))
                    z_cores[k + 1] = einsum('rdc,cbR->rbdR',
                                            (Rz @ (np.diag(sz[:trunc_r]) @ vz[:trunc_r])).reshape(trunc_r, block_size, rz[k + 1]),
                                            z_cores[k + 1])
                    legacy_rz_kp1 = rz[k+1]
                    rz[k + 1] = trunc_r
                else:
                    # lr orthogonalise and shift block left to right
                    z_cores[k] = np.reshape(cz_new, (rz[k], block_size, N[k], rz[k + 1]))
                    legacy_rz_kp1 = rz[k + 1]

            if k < d - 1:
                if not last and amen:
                    # Enrichment
                    left_res = np.zeros((rx[k], N[k], block_size, legacy_rz_kp1))
                    local_core = np.reshape(u @ v, [rx[k], N[k], block_size, rx[k + 1]])
                    for (i, j) in block_A:
                        left_res[:, :, i] += _local_product(ZAX[(i, j)][k + 1], XAX[(i, j)][k], block_A[(i, j)][k], local_core[:, :, j])
                    left_b = np.zeros((rx[k], N[k], block_size, legacy_rz_kp1))
                    for i in block_b:
                        left_b[:, :, i] = einsum('br,bmB,BR->rmR', Xb[i][k], nrmsc[i] * block_b[i][k], Zb[i][k + 1])
                    uk = left_b - left_res  # rx_k x N_k x b x rz_k+1
                    uk = uk.reshape(rx[k]*N[k]*block_size, -1)
                    enriched_u = np.concatenate((np.reshape(u, (rx[k]*N[k]*block_size, -1)), uk), 1)
                    u = enriched_u.reshape(rx[k]*N[k], -1)
                    v = np.concatenate((v, np.zeros((legacy_rz_kp1, rx[k + 1]))), 0)  # r+rz[k+1] x rx[k+1]
                else:
                    u = u.reshape(rx[k]*N[k], -1)
                # u: rx_k*N_k x enriched_rank, Rmat: enriched_rank x block_size*(r+rz[k+1])
                u, Rmat = np.linalg.qr(u)
                r = u.shape[1]
                Rtens = Rmat.reshape(r, block_size, -1)
                v = einsum("rbi, iR, Rdk -> rbdk", Rtens, v, x_cores[k + 1]) #  enriched_r x b x d x rx[k+1]

                nrmsc = nrmsc * normA[:, k] * normx[k] / normb[:, k]
                norm_now = np.linalg.norm(v)
                v /= norm_now
                normx[k] *= norm_now

                x_cores[k] = np.reshape(u, [rx[k], N[k], r])
                x_cores[k + 1] = np.reshape(v, [r, block_size, N[k + 1], rx[k + 2]])
                rx[k + 1] = r
                row_A = np.zeros(block_size)
                for (i, j) in block_A:
                    XAX_ij_k = _compute_phi_fwd_A(XAX[(i, j)][k], x_cores[k], block_A[(i, j)][k], x_cores[k])
                    row_A[i] += np.linalg.norm(XAX_ij_k)**2
                    XAX[(i, j)][k + 1] = XAX_ij_k
                    if not last and amen:
                        ZAX[(i, j)][k + 1] = _compute_phi_fwd_A(ZAX[(i, j)][k], z_cores[k], block_A[(i, j)][k], x_cores[k])
                row_A += (row_A < real_tol)
                normA[:, k] = np.sqrt(row_A)
                for (i, j) in block_A:
                    XAX[(i, j)][k + 1] /= normA[i, k]
                    if not last and amen:
                        ZAX[(i, j)][k + 1] /= normA[i, k]
                for i in block_b:
                    Xb_i_k = _compute_phi_fwd_rhs(Xb[i][k], block_b[i][k], x_cores[k])
                    norm = np.linalg.norm(Xb_i_k)
                    norm = norm if norm > real_tol else 1.0
                    normb[i, k] = norm
                    Xb[i][k + 1] = Xb_i_k / norm
                    if not last and amen:
                        Zb[i][k + 1] = _compute_phi_fwd_rhs(Zb[i][k], block_b[i][k], z_cores[k]) / norm

                nrmsc = nrmsc * normb[:, k] / (normA[:, k] * normx[k])

            else:
                current_core = np.reshape(u @ v, (rx[k], N[k], block_size, rx[k + 1]))
                x_cores[k] = np.transpose(current_core, (0, 2, 1, 3))

        if last:
            break

        if swp >= nswp - 2:
            last = True

        if max_res < eps or max_dx < eps:
            last = True
            if not amen:
                break

    if verbose:
        print("\n\t---Results---")
        print('\tSolution rank is', rx[1:-1])
        print('\tResidual ', max_res)
        print('\tNumber of sweeps', swp+1)
        print('\tTime: ', time.time() - t0)
        print('\tTime per sweep: ', (time.time() - t0) / (swp+1))

    normx = np.exp(np.sum(np.log(normx)) / d)

    return [normx * core for core in x_cores], max_res


def svd_solve_local_system(B, rhs, eps):
    u, s, v = scip.linalg.svd(B, full_matrices=False, check_finite=False)
    # small singular values cause  numerical instabilities
    r = np.sum([s > eps])
    solution_now = v[:r, :].T @ (np.diag(np.divide(1, s[:r])) @ (u[:, :r].T @ rhs))
    return solution_now

def pd_inv(a):
    return scip.linalg.solve(a, np.identity(a.shape[0]), assume_a="pos", overwrite_b=True)

def  ipm_solve_local_system(lhs, rhs, block_dim, num_blocks, eps):
    k =  num_blocks - 1
    L_eq = -lhs[block_dim:2*block_dim, :block_dim]
    L_Z = lhs[k*block_dim:, :block_dim]
    inv_L_Z = pd_inv(L_Z)
    L_eq_adj = -lhs[:block_dim, block_dim:2*block_dim]
    #I = lhs[:block_dim, k*block_dim:]
    inv_I = np.diag(np.divide(1, np.diagonal(lhs[:block_dim, k*block_dim:])))
    L_X = lhs[k * block_dim:, k * block_dim:]
    R_d = -rhs[:block_dim]
    R_p = -rhs[block_dim:2*block_dim]
    R_c = -rhs[k * block_dim:]
    if num_blocks > 3:
        TL_ineq = -lhs[2 * block_dim:3 * block_dim, :block_dim]
        L_ineq_adj = -lhs[:block_dim, 2 * block_dim:3 * block_dim]
        R_ineq = lhs[2 * block_dim:3 * block_dim, 2 * block_dim:3 * block_dim]
        R_t = -rhs[2 * block_dim:3 * block_dim]
        A = np.block([
            [L_eq @ inv_L_Z @ L_X @ inv_I @ L_eq_adj, L_eq @ inv_L_Z @ L_X @ inv_I @ L_ineq_adj],
            [TL_ineq @ inv_L_Z @ L_X @ inv_I @ L_eq_adj, R_ineq + TL_ineq @ inv_L_Z @ L_X @ inv_I @ L_ineq_adj]
        ])
        b = np.block([
            [L_eq @ inv_L_Z @ (L_X @ inv_I @ R_d - R_c) - R_p],
            [TL_ineq @ inv_L_Z @ (L_X @ inv_I @ R_d - R_c) - R_t]
        ])
        yt, _, _, _ = scip.linalg.lstsq(A, b, cond=eps, check_finite=False)
        y = yt[:block_dim]
        t = yt[block_dim:]
        x = inv_L_Z @ (L_X @ inv_I @ (R_d - L_eq_adj @ y - L_ineq_adj @ t) - R_c)
        z = inv_I @ (L_eq_adj @ y + L_ineq_adj @ t - R_d)
        return np.vstack((x, y, t, z))
    #print(inv_L_Z)
    y, _, _, _ = scip.linalg.lstsq(L_eq @ inv_L_Z @ L_X @ inv_I @ L_eq_adj, L_eq @ inv_L_Z @ (L_X @ inv_I @ R_d - R_c) - R_p, cond=eps, check_finite=False)
    x = inv_L_Z @ (L_X @ inv_I @ (R_d - L_eq_adj @ y) - R_c)
    z = inv_I @ (L_eq_adj @ y - R_d)
    #print("---")
    #print(np.linalg.norm(- L_eq_adj @ y + I @ z + R_d))
    #print(np.linalg.norm(- L_eq @  x + R_p))
    #print(np.linalg.norm(L_Z @ x + L_X @ z + R_c))
    #print("---")
    return np.vstack((x, y, z))


def schur_solve_local_system(lhs, rhs, block_dim, num_blocks, eps):
    k =  num_blocks - 1
    A = lhs[:k * block_dim, :k * block_dim]
    B = lhs[:k * block_dim, k * block_dim:]
    C = lhs[k * block_dim:, :k * block_dim]
    D = lhs[k * block_dim:, k * block_dim:]
    u = rhs[:k * block_dim]
    v = rhs[k * block_dim:]
    inv_D = np.linalg.inv(D)
    schur_complement = A - B @ inv_D @ C
    x, _, _, _ = scip.linalg.lstsq(schur_complement, u-B @ (inv_D @ v), cond=eps, check_finite=False) # l1_lstq(schur_complement, u-B @ (inv_D @ v))
    y = inv_D @ (v - C @ x)
    return np.vstack((x, y))

# L1-regularized least squares cost function
def l1_regularized_least_squares(w, X, y, lam):
    residual = y - X @ w
    return np.linalg.norm(residual) + lam * np.sum(np.abs(w))

def l1_lstq(lhs, rhs):
    # Initial guess
    w0 = np.zeros(lhs.shape[1])
    lam = 0.01  # Regularization parameter
    # Solve using L-BFGS-B (or try other methods like 'trust-constr')
    result = scip.optimize.minimize(l1_regularized_least_squares, w0, args=(lhs, rhs, lam))
    return result.x.reshape(-1,  1)




