import copy
import sys
import os
import time

sys.path.append(os.getcwd() + '/../')

from src.tt_ops import *
from cy_src.ops_cy import *
from src.tt_amen import _compute_phi_bck_A, cached_einsum

def _step_size_local_solve(previous_solution, XDX_k, Delta_k, XDX_k1, XAX_k, A_k, XAX_k1, m, step_size, size_limit, eps):
    if m <= size_limit:
        previous_solution = previous_solution.reshape(-1, 1)
        D = cached_einsum(
            "lsr,smnS,LSR->lmLrnR",
            XDX_k, Delta_k, XDX_k1
        ).reshape(m, m)
        A = cached_einsum("lsr,smnS,LSR->lmLrnR", XAX_k, A_k, XAX_k1).reshape(m, m)
        try:
            eig_val, solution_now = scip.sparse.linalg.eigsh((1/step_size)*A + D, tol=eps, k=1, which="SA", v0=previous_solution)
        except:
            eig_val = previous_solution.T @ ((1/step_size)*A + D)  @ previous_solution
            solution_now = previous_solution
        if eig_val < 0:
            try:
                eig_val, solution_now = scip.sparse.linalg.eigsh(-D, M=A, tol=eps, k=1, which="LA", v0=previous_solution)
                step_size = max(0, min(step_size, 1/ eig_val[0]))
            except:
                solution_now = previous_solution

        eig_val = previous_solution.T @ ((1/step_size)*A + D) @ previous_solution
        old_res = np.linalg.norm(((1/step_size)*A + D) @ previous_solution - eig_val*previous_solution)
    else:
        x_shape = previous_solution.shape
        previous_solution = previous_solution.reshape(-1, 1)
        # 'lsr,smnk,LSR,rnR-> lmkLS' 'ks'
        _mat_vec_A = contract_expression('lsr,smnS,LSR,rnR->lmL', XAX_k.shape, A_k.shape, XAX_k1.shape, x_shape, optimize="greedy")
        mat_vec_A = lambda x_vec: _mat_vec_A(XAX_k, A_k, XAX_k1, x_vec.reshape(*x_shape)).reshape(-1, 1)
        A_op = scip.sparse.linalg.LinearOperator((m, m), matvec=mat_vec_A)
        _mat_vec_D = contract_expression('lsr,smnS,LSR,rnR->lmL', XDX_k.shape, Delta_k.shape, XDX_k1.shape, x_shape, optimize="greedy")
        mat_vec_D = lambda x_vec: -_mat_vec_D(XDX_k, Delta_k, XDX_k1, x_vec.reshape(*x_shape)).reshape(-1, 1)
        D_op = scip.sparse.linalg.LinearOperator((m, m), matvec=mat_vec_D)
        AD_op = scip.sparse.linalg.LinearOperator((m, m), matvec=lambda x_vec: mat_vec_A(x_vec) / step_size - mat_vec_D(x_vec))

        try:
            eig_val, solution_now = scip.sparse.linalg.eigsh(AD_op, tol=eps, k=1, which="SA", v0=previous_solution)
        except:
            eig_val = previous_solution.T @ AD_op(previous_solution)
            solution_now = previous_solution
        if eig_val < 0:
            try:
                eig_val, solution_now = scip.sparse.linalg.eigsh(D_op, M=A_op, tol=eps, k=1, which="LA", v0=previous_solution)
                step_size = max(0, min(step_size, 1 / eig_val[0]))
            except:
                solution_now = previous_solution

        eig_val = previous_solution.T @ AD_op(previous_solution)
        old_res = np.linalg.norm(AD_op(previous_solution) - eig_val * previous_solution)

    return solution_now.reshape(-1, 1), step_size, old_res


def _local_psd_check(previous_solution, XAX_k, A_k, XAX_k1, m, size_limit, eps):
    if m <= size_limit:
        try:
            eig_val, _ = scip.sparse.linalg.eigsh(cached_einsum("lsr,smnS,LSR->lmLrnR", XAX_k, A_k, XAX_k1).reshape(m, m), tol=eps, k=1, which="SA")
        except:
            eig_val = -1
    else:
        x_shape = previous_solution.shape
        _mat_vec_A = contract_expression('lsr,smnS,LSR,rnR->lmL', XAX_k.shape, A_k.shape, XAX_k1.shape, x_shape, optimize="greedy")
        mat_vec_A = lambda x_vec: _mat_vec_A(XAX_k, A_k, XAX_k1, x_vec.reshape(*x_shape)).reshape(-1, 1)
        A_op = scip.sparse.linalg.LinearOperator((m, m), matvec=mat_vec_A)
        try:
            eig_val, _ = scip.sparse.linalg.eigsh(A_op, tol=eps, k=1, which="SA")
        except:
            eig_val = -1

    return eig_val >= 0


def _add_kick_rank(u, v, r_add=2):
    old_r = u.shape[-1]
    uk = np.random.randn(u.shape[0], r_add)  # rx_k x N_k x rz_k+1
    u, Rmat = scp.linalg.qr(np.concatenate((u, uk), 1), check_finite=False, mode="economic", overwrite_a=True)
    v = Rmat[:, :old_r] @ v
    return u, v, u.shape[-1]


def tt_max_generalised_eigen(A, Delta, x0=None, kick_rank=None, nswp=10, tol=1e-12, verbose=False):
    if verbose:
        print(f"Starting Eigen solve with:\n \t {tol} \n \t sweeps: {nswp}")
        t0 = time.time()
    if x0 is None:
        x_cores = tt_random_gaussian([2]*(len(A)-1), (A[0].shape[2],))
    else:
        x_cores = x0
    if kick_rank is None:
        kick_rank = np.maximum(np.ceil(symmetric_powers_of_two(len(A))/nswp), 2).astype(int)

    d = len(x_cores)
    rx = np.array([1] + tt_ranks(x_cores) + [1])
    N = np.array([c.shape[1] for c in x_cores])

    XAX = [np.ones((1, 1, 1))] + [None] * (d - 1) + [np.ones((1, 1, 1))]  # size is rk x Rk x rk
    XDX = copy.deepcopy(XAX)

    step_size = 1
    last = False
    size_limit = N[0] * (int(np.sqrt(d) * d))**2 / (d/2)
    local_res = np.inf*np.ones((2, d-1))
    for swp in range(nswp):
        max_res = np.inf if swp == 0 else 0
        for k in range(d - 1, -1, -1):
            if swp > 0:
                previous_solution = x_cores[k]
                solution_now, step_size, res = _step_size_local_solve(previous_solution, XDX[k], Delta[k], XDX[k+1], XAX[k], A[k], XAX[k+1], rx[k] * N[k] * rx[k + 1], step_size, size_limit, tol)

                if 2*local_res[0, k-1] < res:
                    if not _local_psd_check(previous_solution, XAX[k], A[k], XAX[k+1], rx[k] * N[k] * rx[k + 1], size_limit, tol):
                        break
                else:
                    local_res[0, k-1] = res
                solution_now = np.reshape(solution_now, (rx[k], N[k] * rx[k + 1])).T
            else:
                solution_now = np.reshape(x_cores[k], (rx[k], N[k] * rx[k + 1])).T

            if k > 0:
                u, s, v = scip.linalg.svd(solution_now, full_matrices=False, check_finite=False, overwrite_a=True, lapack_driver="gesvd")
                v = s.reshape(-1, 1) * v
                r = prune_singular_vals(s, 0.5*tol)
                if not last:
                    kick = kick_rank[k]
                    u, v, r = _add_kick_rank(u[:, :r], v[:r], kick)
                else:
                    u = u[:, :r]
                    v = v[:r]
                x_cores[k] = np.reshape(u.T, (r, N[k], rx[k + 1]))
                x_cores[k - 1] = einsum('rdc,cR->rdR', x_cores[k - 1], v.T, optimize=[(0, 1)])
                rx[k] = r

                XAX[k] = _compute_phi_bck_A(XAX[k + 1], x_cores[k], A[k], x_cores[k])
                XDX[k] = _compute_phi_bck_A(XDX[k + 1], x_cores[k], Delta[k], x_cores[k])
                norm = np.sqrt(np.linalg.norm(XAX[k]) ** 2 + np.linalg.norm(XDX[k]) ** 2)
                norm = norm if norm > 0 else 1.0
                XAX[k] /= norm
                XDX[k] /= norm

            else:
                x_cores[k] = np.reshape(solution_now, (rx[k], N[k], rx[k + 1]))

        x_cores = tt_normalise(x_cores)
        if last:
            break
        if np.max(local_res[0]) < tol or swp == nswp - 1:
            last = True
        if verbose:
            print('\tStarting Sweep: %d' % swp)
            print('\tStep size: %f' % step_size)
            print(f"\tDirection: {-1}")
            print(f'\tResidual {max_res}')
            print(f"\tTT-sol rank: {tt_ranks(x_cores)}", flush=True)
        max_res = 0
        for k in range(d):
            previous_solution = x_cores[k]
            solution_now, step_size, res = _step_size_local_solve(previous_solution, XDX[k], Delta[k], XDX[k+1], XAX[k], A[k], XAX[k+1], rx[k] * N[k] * rx[k + 1], step_size, size_limit, tol)
            if 2*local_res[1, k-1] < res:
                if not _local_psd_check(previous_solution, XAX[k], A[k], XAX[k + 1], rx[k] * N[k] * rx[k + 1], size_limit, tol):
                    break
            else:
                local_res[1, k-1] = res
            solution_now = np.reshape(solution_now, (rx[k] * N[k], rx[k + 1]))
            if k < d - 1:
                u, s, v = scip.linalg.svd(solution_now, full_matrices=False, check_finite=False, overwrite_a=True, lapack_driver="gesvd")
                v = s.reshape(-1, 1) * v
                r = prune_singular_vals(s, 0.5*tol)
                if not last:
                    kick = kick_rank[k]
                    u, v, r = _add_kick_rank(u[:, :r], v[:r, :], kick)
                else:
                    u = u[:, :r]
                    v = v[:r, :]
                x_cores[k] = u.reshape(rx[k], N[k], r)
                x_cores[k + 1] = einsum('ij,jkl->ikl', v, x_cores[k + 1], optimize=[(0, 1)]).reshape(r, N[k + 1], rx[k + 2])
                rx[k + 1] = r
                XAX[k + 1] = compute_phi_fwd_A(XAX[k], x_cores[k], A[k], x_cores[k])
                XDX[k + 1] = compute_phi_fwd_A(XDX[k], x_cores[k], Delta[k], x_cores[k])
                norm = np.sqrt(np.linalg.norm(XAX[k + 1]) ** 2 + np.linalg.norm(XDX[k + 1]) ** 2)
                norm = norm if np.greater(norm, 0) else 1.0
                XAX[k + 1] /= norm
                XDX[k + 1] /= norm
            else:
                x_cores[k] = np.reshape(solution_now, (rx[k], N[k], rx[k + 1]))

        x_cores = tt_normalise(x_cores)
        if last:
            break
        if np.max(local_res[1]) < tol:
            last = True
        if verbose:
            print('\tStarting Sweep: %d' % swp)
            print('\tStep size: %f' % step_size)
            print(f"\tDirection: {1}")
            print(f'\tResidual {max_res}')
            print(f"\tTT-sol rank: {tt_ranks(x_cores)}", flush=True)

    if verbose:
        print("\t -----")
        print(f"\t Solution rank is {rx[1:-1]}")
        print('\t Step size: %f' % step_size)
        print(f"\t Residual {max_res}")
        print('\t Number of sweeps', swp + 1)
        print('\t Time: ', time.time() - t0)
        print('\t Time per sweep: ', (time.time() - t0) / (swp + 1))

    if max_res > tol:
        step_size = 0
    return step_size


def tt_min_eig(A, x0=None, kick_rank=None, nswp=10, tol=1e-12, verbose=False):
    if verbose:
        print(f"Starting Eigen solve with:\n \t {tol} \n \t sweeps: {nswp}")
        t0 = time.time()
    if x0 is None:
        x_cores = tt_random_gaussian([2]*(len(A)-1), (A[0].shape[2],))
    else:
        x_cores = x0
    if kick_rank is None:
        kick_rank = np.maximum((symmetric_powers_of_two(len(A))/(nswp -1)), 2).astype(int)
    d = len(x_cores)
    rx = np.array([1] + tt_ranks(x_cores) + [1])
    N = np.array([c.shape[1] for c in x_cores])

    XAX = [np.ones((1, 1, 1))] + [None] * (d - 1) + [np.ones((1, 1, 1))]  # size is rk x Rk x rk

    max_res = 0
    last = False
    size_limit = N[0] * (int(np.sqrt(d) * d))**2 / (d/2)
    for swp in range(nswp):
        max_res = np.inf if swp == 0 else 0
        for k in range(d - 1, -1, -1):
            if swp > 0:
                previous_solution = x_cores[k]
                solution_now, local_res = _eigen_local_solve(previous_solution, XAX[k], A[k], XAX[k+1], rx[k] * N[k] * rx[k + 1], size_limit, tol)
                max_res = max(max_res, local_res)
                solution_now = np.reshape(solution_now, (rx[k], N[k] * rx[k + 1])).T
            else:
                solution_now = np.reshape(x_cores[k], (rx[k], N[k] * rx[k + 1])).T

            if k > 0:
                u, s, v = scip.linalg.svd(solution_now, full_matrices=False, check_finite=False, overwrite_a=True, lapack_driver="gesvd")
                v = s.reshape(-1, 1) * v
                r = prune_singular_vals(s, 0.5*tol)
                if not last:
                    kick = kick_rank[k]
                    u, v, r = _add_kick_rank(u[:, :r], v[:r], kick)
                else:
                    u = u[:, :r]
                    v = v[:r]
                x_cores[k] = np.reshape(u.T, (r, N[k], rx[k + 1]))
                x_cores[k - 1] = einsum('rdc,cR->rdR', x_cores[k - 1], v.T, optimize=[(0, 1)])
                rx[k] = r

                XAX[k] = _compute_phi_bck_A(XAX[k + 1], x_cores[k], A[k], x_cores[k])
                norm = np.linalg.norm(XAX[k])
                norm = norm if norm > 0 else 1.0
                XAX[k] /= norm
            else:
                x_cores[k] = np.reshape(solution_now, (rx[k], N[k], rx[k + 1]))

        x_cores = tt_normalise(x_cores)
        if last:
            break
        if max_res < tol or swp == nswp - 1:
            last = True
        if verbose:
            print('\tStarting Sweep: %d' % swp)
            print(f"\tDirection: {-1}")
            print(f'\tResidual {max_res}')
            print(f"\tTT-sol rank: {tt_ranks(x_cores)}", flush=True)
        max_res = 0
        for k in range(d):
            previous_solution = x_cores[k]
            solution_now, local_res = _eigen_local_solve(previous_solution, XAX[k], A[k], XAX[k+1], rx[k] * N[k] * rx[k + 1], size_limit, tol)
            max_res = max(max_res, local_res)
            solution_now = np.reshape(solution_now, (rx[k] * N[k], rx[k + 1]))
            if k < d - 1:
                u, s, v = scip.linalg.svd(solution_now, full_matrices=False, check_finite=False, overwrite_a=True, lapack_driver="gesvd")
                v = s.reshape(-1, 1) * v
                r = prune_singular_vals(s, 0.5*tol)
                if not last:
                    kick = kick_rank[k]
                    u, v, r = _add_kick_rank(u[:, :r], v[:r, :], kick)
                else:
                    u = u[:, :r]
                    v = v[:r, :]
                x_cores[k] = u.reshape(rx[k], N[k], r)
                x_cores[k + 1] = einsum('ij,jkl->ikl', v, x_cores[k + 1], optimize=[(0, 1)]).reshape(r, N[k + 1], rx[k + 2])
                rx[k + 1] = r
                XAX[k + 1] = compute_phi_fwd_A(XAX[k], x_cores[k], A[k], x_cores[k])
                norm = np.linalg.norm(XAX[k + 1])
                norm = norm if np.greater(norm, 0) else 1.0
                XAX[k + 1] /= norm
            else:
                x_cores[k] = np.reshape(solution_now, (rx[k], N[k], rx[k + 1]))

        x_cores = tt_normalise(x_cores)
        if last:
            break
        if max_res < tol:
            last = True
        if verbose:
            print('\tStarting Sweep: %d' % swp)
            print(f"\tDirection: {1}")
            print(f'\tResidual {max_res}')
            print(f"\tTT-sol rank: {tt_ranks(x_cores)}", flush=True)

    if verbose:
        print("\t -----")
        print(f"\t Solution rank is {rx[1:-1]}")
        print(f"\t Residual {max_res}")
        print('\t Number of sweeps', swp + 1)
        print('\t Time: ', time.time() - t0)
        print('\t Time per sweep: ', (time.time() - t0) / (swp + 1))

    min_eig_value = tt_inner_prod(x_cores, tt_fast_matrix_vec_mul(A, x_cores, tol)).squeeze()
    return x_cores, min_eig_value


def _eigen_local_solve(previous_solution, XAX_k, A_k, XAX_k1, m, size_limit, eps):
    if m <= size_limit:
        previous_solution = previous_solution.reshape(-1, 1)
        A = cached_einsum("lsr,smnS,LSR->lmLrnR", XAX_k, A_k, XAX_k1).reshape(m, m)
        try:
            eig_val, solution_now = scip.sparse.linalg.eigsh(A, tol=eps, k=1, which="SA", v0=previous_solution)
        except:
            solution_now = previous_solution
            eig_val = previous_solution.T @ A @ previous_solution
        old_res = np.linalg.norm(eig_val * previous_solution - A @ previous_solution)
        return solution_now, old_res

    x_shape = previous_solution.shape
    previous_solution = previous_solution.reshape(-1, 1)
    _mat_vec_A = contract_expression('lsr,smnS,LSR,rnR->lmL', XAX_k.shape, A_k.shape, XAX_k1.shape, x_shape, optimize="greedy")
    mat_vec_A = lambda x_vec: _mat_vec_A(XAX_k, A_k, XAX_k1, x_vec.reshape(*x_shape)).reshape(-1, 1)
    A_op = scip.sparse.linalg.LinearOperator((m, m), matvec=mat_vec_A)
    try:
        eig_val, solution_now = scip.sparse.linalg.eigsh(A_op, tol=eps, k=1, which="SA", v0=previous_solution)
    except:
        solution_now = previous_solution
        eig_val = previous_solution.T @ A_op(previous_solution)

    old_res = np.linalg.norm(eig_val * previous_solution - A_op(previous_solution))

    return solution_now.reshape(-1, 1), old_res


def tt_elementwise_max(vec_tt, val, nswp=4, eps=1e-10, verbose=False):
    if verbose:
        print(f"Starting Eigen solve with:\n \t {eps} \n \t sweeps: {nswp}")
        t0 = time.time()
    vec_tt_norm = np.sqrt(tt_inner_prod(vec_tt, vec_tt))
    A = tt_scale(np.divide(1, vec_tt_norm), tt_diag(vec_tt))
    if val != 0:
        A = tt_sub(A, tt_scale(val+eps, tt_identity(len(A))))
    A = tt_rank_reduce(A, eps=eps)
    dtype = A[0].dtype
    x_cores = [np.ones_like(c[:, :, 0], dtype=dtype) for c in A]

    d = len(x_cores)
    rx = np.array([1] + tt_ranks(x_cores) + [1])
    N = np.array([c.shape[1] for c in x_cores])

    XAX = [np.ones((1, 1, 1), dtype=dtype)] + [None] * (d - 1) + [np.ones((1, 1, 1), dtype=dtype)]  # size is rk x Rk x rk

    max_res = 0
    real_tol = (eps / np.sqrt(d))
    all_negative = False
    for swp in range(nswp):
        x_cores = tt_rl_orthogonalise(x_cores)
        rx[1:-1] = np.array(tt_ranks(x_cores))
        XAX, no = compute_phi_bcks_A(XAX, x_cores, A, x_cores, d=d)

        # start loop
        max_res = 0

        for k in range(d):

            # solve the local system
            Bp = cached_einsum("smnS,LSR->smnRL", A[k], XAX[k + 1])
            B = cached_einsum("lsr,smnRL->lmLrnR", XAX[k], Bp)
            B = np.reshape(B, [rx[k] * N[k] * rx[k + 1], rx[k] * N[k] * rx[k + 1]])

            #TODO: Need to normalise x_cores ?
            eig_vals, Q = scip.linalg.eigh(B, check_finite=False)
            l = np.sum(eig_vals > -eps)
            if l == 0:
                all_negative = True
                x_cores = [np.zeros((1, 2, 1)) for _ in x_cores]
                break
            solution_now = Q[:, -l:]

            b = solution_now.shape[-1]
            solution_now = np.reshape(solution_now, (rx[k] * N[k], b*rx[k+1]))

            # solution truncation
            norm = np.linalg.norm(solution_now)
            if k < d - 1 and False:
                # FIXME: We need to do svd on (rx*N) x (block_size*rx), otherwise the pruning is ineffective
                u, s, v = scip.linalg.svd(solution_now, full_matrices=False, check_finite=False, lapack_driver="gesvd")
                v = np.diag(s) @ v
                r = 0
                for r in range(u.shape[1] - 1, 0, -1):
                    solution = u[:, :r] @ v[:r, :]
                    res = np.linalg.norm(solution - solution_now) / norm
                    if res > real_tol:
                        break
                r += 1
                u = u[:, :r]
                v = v[:r, :]
            else:
                u, v = np.linalg.qr(solution_now)
                r = u.shape[1]
            u = u[:, :r]
            v = v[:r, :]


            if k < d - 1:
                v = cached_einsum('ij,jkl->ikl', v, np.tile(x_cores[k + 1], (b, 1, 1)))

                x_cores[k] = np.reshape(u, [rx[k], N[k], r])
                x_cores[k + 1] = np.reshape(v, [r, N[k + 1], rx[k + 2]])
                rx[k + 1] = r

                # next phis with norm correction
                XAX[k + 1] = compute_phi_fwd_A(XAX[k], x_cores[k], A[k], x_cores[k])

                # ... and norms
                norm = np.linalg.norm(XAX[k + 1])
                norm = norm if np.greater(norm, 0) else 1.0
                XAX[k + 1] /= norm

            else:
                x_cores[k] = np.reshape(np.tile(u,  (1, b)) @ v.reshape(r*b, rx[k + 1]), (rx[k], N[k], rx[k + 1]))

        if all_negative:
            break

    if verbose:
        print("\t -----")
        print(f"\t Solution rank is {rx[1:-1]}")
        print(f"\t Residual {max_res}")
        print('\t Number of sweeps', swp + 1)
        print('\t Time: ', time.time() - t0)
        print('\t Time per sweep: ', (time.time() - t0) / (swp + 1))


    x_cores = tt_fast_hadammard(x_cores, x_cores, eps)

    vec_tt = tt_scale(vec_tt_norm, tt_fast_matrix_vec_mul(A, x_cores, eps))
    if val != 0:
        vec_tt = tt_rank_reduce(tt_add(vec_tt, tt_scale(val + eps, [np.ones((1, 2, 1)) for _ in vec_tt])), eps)

    return vec_tt

def tt_elementwise_min(vec_tt, val, nswp=4, eps=1e-10, verbose=False):
    return tt_scale(-1, tt_elementwise_max(tt_scale(-1, vec_tt), -val, nswp, eps, verbose))


def symmetric_powers_of_two(length):
    half = length // 2
    first_half = [2**i for i in range(1, half + 1)]
    if length % 2 == 0:
        return np.array(first_half + first_half[::-1])
    else:
        return np.array(first_half + [2**(half + 1)] + first_half[::-1])


def tt_approx_mat_mat_mul(A, D, x0=None, kick_rank=None, nswp=20, tol=1e-6, verbose=False):
    if verbose:
        print(f"Starting Eigen solve with:\n \t {tol} \n \t sweeps: {nswp}")
        t0 = time.time()
    if x0 is None:
        x_cores = tt_random_gaussian([2]*(len(A)-1), A[0].shape[1:-1])
    else:
        x_cores = x0

    if kick_rank is None:
        kick_rank = np.maximum((symmetric_powers_of_two(len(A))/(nswp -1)), 2).astype(int)

    d = len(x_cores)
    rx = np.array([1] + tt_ranks(x_cores) + [1])
    N = np.array([c.shape[1] for c in x_cores])
    M = np.array([c.shape[2] for c in x_cores])

    XADX = [np.ones((1, 1, 1))] + [None] * (d - 1) + [np.ones((1, 1, 1))]  # size is rk x Rk x rk

    normAD = np.ones(d - 1)  # norm of each row in the block matrix
    nrmsc = 1.0
    normx = np.ones((d - 1))

    max_res = 0
    last = False
    for swp in range(nswp):
        max_res = np.inf if swp == 0 else 0
        for k in range(d - 1, -1, -1):
            if swp > 0:
                previous_solution = x_cores[k]
                solution_now = cached_einsum('rab,amkA,bknB,RAB->rmnR',XADX[k], A[k], D[k], XADX[k+1])
                solution_now *= nrmsc
                local_res = np.linalg.norm(solution_now - previous_solution) / np.linalg.norm(solution_now)
                max_res = max(max_res, local_res)
                solution_now = np.reshape(solution_now, (rx[k], N[k] * M[k] * rx[k + 1])).T
            else:
                solution_now = np.reshape(x_cores[k], (rx[k], N[k] * M[k] * rx[k + 1])).T

            if k > 0:
                u, s, v = scip.linalg.svd(solution_now, full_matrices=False, check_finite=False, overwrite_a=True, lapack_driver="gesvd")
                v = s.reshape(-1, 1) * v
                r = prune_singular_vals(s, 0.5*tol)
                if not last:
                    kick = kick_rank[k]
                    u, v, r = _add_kick_rank(u[:, :r], v[:r], kick)
                else:
                    u = u[:, :r]
                    v = v[:r]
                nrmsc *= normx[k - 1] / normAD[k - 1]
                x_cores[k] = np.reshape(u.T, (r, N[k], M[k], rx[k + 1]))
                x_cores[k - 1] = einsum('rdkc,cR->rdkR', x_cores[k - 1], v.T, optimize=[(0, 1)])
                norm_now = np.linalg.norm(x_cores[k - 1])
                normx[k - 1] *= norm_now
                x_cores[k - 1] /= norm_now
                rx[k] = r

                XADX[k] = cached_einsum('RAB,amkA,bknB,rmnR->rab', XADX[k+1], A[k], D[k], x_cores[k])
                norm = np.linalg.norm(XADX[k])
                norm = norm if norm > 0 else 1.0
                XADX[k] /= norm
                normAD[k-1] = norm
                nrmsc *= normAD[k - 1] / normx[k - 1]
            else:
                x_cores[k] = np.reshape(solution_now, (rx[k], N[k], M[k], rx[k + 1]))

        if last:
            break
        if max_res < tol or swp == nswp - 1:
            last = True
        if verbose:
            print('\tStarting Sweep: %d' % swp)
            print(f"\tDirection: {-1}")
            print(f'\tResidual {max_res}')
            print(f"\tTT-sol rank: {tt_ranks(x_cores)}", flush=True)
        max_res = 0
        for k in range(d):
            previous_solution = x_cores[k]
            solution_now = cached_einsum('rab,amkA,bknB,RAB->rmnR', XADX[k], A[k], D[k], XADX[k + 1])
            solution_now *= nrmsc
            local_res = np.linalg.norm(solution_now - previous_solution) / np.linalg.norm(solution_now)
            max_res = max(max_res, local_res)
            solution_now = np.reshape(solution_now, (rx[k] * N[k] * M[k], rx[k + 1]))
            if k < d - 1:
                nrmsc *= normx[k] / normAD[k]
                u, s, v = scip.linalg.svd(solution_now, full_matrices=False, check_finite=False, overwrite_a=True, lapack_driver="gesvd")
                v = s.reshape(-1, 1) * v
                r = prune_singular_vals(s, 0.5*tol)
                if not last:
                    kick = kick_rank[k]
                    u, v, r = _add_kick_rank(u[:, :r], v[:r, :], kick)
                else:
                    u = u[:, :r]
                    v = v[:r, :]
                x_cores[k] = u.reshape(rx[k], N[k], M[k], r)
                x_cores[k + 1] = einsum('ij,jdkl->idkl', v, x_cores[k + 1], optimize=[(0, 1)]).reshape(r, N[k + 1], M[k+1], rx[k + 2])
                norm_now = np.linalg.norm(x_cores[k + 1])
                normx[k] *= norm_now
                x_cores[k + 1] /= norm_now
                rx[k + 1] = r

                XADX[k + 1] = cached_einsum('rab,amkA,bknB,rmnR->RAB', XADX[k], A[k], D[k], x_cores[k])
                norm = np.linalg.norm(XADX[k + 1])
                norm = norm if np.greater(norm, 0) else 1.0
                XADX[k + 1] /= norm
                normAD[k] = norm
                nrmsc *= normAD[k] / normx[k]
            else:
                x_cores[k] = np.reshape(solution_now, (rx[k], N[k], M[k], rx[k + 1]))

        if last:
            break
        if max_res < tol:
            last = True
        if verbose:
            print('\tStarting Sweep: %d' % swp)
            print(f"\tDirection: {1}")
            print(f'\tResidual {max_res}')
            print(f"\tTT-sol rank: {tt_ranks(x_cores)}", flush=True)

    if verbose:
        print("\t -----")
        print(f"\t Solution rank is {rx[1:-1]}")
        print(f"\t Residual {max_res}")
        print('\t Number of sweeps', swp + 1)
        print('\t Time: ', time.time() - t0)
        print('\t Time per sweep: ', (time.time() - t0) / (swp + 1))

    normx = np.exp(np.sum(np.log(normx)) / d)

    return [normx * core for core in x_cores]


def tt_mat_mat_mul(mat1, mat2, op_tol, eps):
    if np.max(np.array(tt_ranks(mat1))*np.array(tt_ranks(mat2))) <= 2**(len(mat1)-1):
        return tt_rank_reduce(tt_fast_mat_mat_mul(mat1, mat2, eps), eps=op_tol)
    return tt_approx_mat_mat_mul(mat1, mat2, tol=op_tol)