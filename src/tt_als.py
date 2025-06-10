import sys
import os
import time

sys.path.append(os.getcwd() + '/../')

from src.tt_ops import *
from opt_einsum import contract as einsum

class TTBlockVector:
    def __init__(self):
        self._data = {}  # maps row index -> list

    def __setitem__(self, index, value):
        if not isinstance(value, list):
            raise ValueError("Each entry must be a list")
        self._data[index] = value

    def get_row(self, index):
        """Return the list at index `index`."""
        return self._data.get(index, None)

    def __getitem__(self, list_index):
        """Returns a view over all rows at a fixed list index."""
        return TTBlockVectorView(self._data, list_index)

    def __iter__(self):
        return iter(self._data)

    def keys(self):
        return self._data.keys()

    def values(self):
        return self._data.values()

    def __repr__(self):
        return repr(self._data)

    @property
    def norm(self):
        return np.sqrt(sum(tt_inner_prod(v, v) for v in self._data.values()))

    def scale(self, s):
        self._data = {key: tt_scale(1/s, value) for (key, value) in self._data.items()}

    def __sub__(self, other):
        block_vec = TTBlockVector()
        for i in self._data.keys():
            block_vec[i] = tt_rank_reduce(tt_sub(self.get_row(i), other.get_row(i)), 1e-12)
        return block_vec



class TTBlockVectorView:
    def __init__(self, data, list_index):
        self._data = data
        self._list_index = list_index

    def __getitem__(self, row_index):
        return self._data[row_index][self._list_index]

    def items(self):
        """Yield (row_index, value) for rows where the value exists."""
        for i, row in self._data.items():
            if self._list_index < len(row):
                yield (i, row[self._list_index])

    def __repr__(self):
        return repr(dict(self.items()))

    def __iter__(self):
        return iter(self._data)

def _tt_get_block(i, block_matrix_tt):
    b = np.argmax([len(c.shape) for c in block_matrix_tt])
    return block_matrix_tt[:b] + [block_matrix_tt[b][:, i]] + block_matrix_tt[b+1:]

class TTBlockMatrix:
    def __init__(self):
        self._data = {}
        self._aliases = {}
        self._transposes = {}


    def add_alias(self, key1, key2, is_transpose=False):
        if is_transpose:
            self._transposes[key1] = key2
        else:
            self._aliases[key1] = key2

    def __getitem__(self, key):
        # Access by (row, col)
        if isinstance(key, tuple) and len(key) == 2:
            return self._data.setdefault(key, [])
        # Access by just list_index → return view object
        elif isinstance(key, int):
            return TTBlockMatrixView(self._data, self._aliases, self._transposes, key)
        else:
            raise KeyError(f"Invalid key format: {key}")

    def __setitem__(self, key, value):
        if isinstance(key, tuple) and len(key) == 2:
            self._data[key] = value
        else:
            raise KeyError(f"Invalid key format: {key}")

    def __repr__(self):
        return f"{self.__class__.__name__}({self._data})"

    def keys(self):
        return self._data.keys()

    def tkeys(self):
        return self._data.keys() | self._transposes.values()

    def akeys(self):
        return self._data.keys() | self._aliases.values()

    def all_keys(self):
        return self._data.keys() | set(self._aliases.values()) | set(self._transposes.values())

    def __iter__(self):
        return iter(self._data)

    def get_submatrix(self, row_index, col_index):
        data = {(i, j): v for (i, j), v in self._data.items() if i <= row_index and j <= col_index}
        aliases = {(i, j):  (k, t) for (i, j), (k, t) in self._aliases.items() if k <= row_index and t <= col_index}
        transposes = {(i, j):  (k, t) for (i, j), (k, t) in self._transposes.items() if k <= row_index and t <= col_index}
        return TTBlockSubMatrix(data, aliases, transposes)


    def matvec(self, x_cores, eps=1e-12):
        result = TTBlockVector()
        for (i, j) in self._data.keys():
            if i in result.keys():
                result[i] = tt_rank_reduce(tt_add(result.get_row(i), tt_fast_matrix_vec_mul(self._data[i, j], _tt_get_block(j, x_cores), eps)), eps)
            else:
                result[i] = tt_fast_matrix_vec_mul(self._data[i, j], _tt_get_block(j, x_cores), eps)
            if (i, j) in self._transposes:
                k, t = self._transposes[i, j]
                if k in result.keys():
                    result[k] = tt_rank_reduce(
                        tt_add(result.get_row(k), tt_fast_matrix_vec_mul(tt_transpose(self._data[i, j]), _tt_get_block(t, x_cores), eps)),
                        eps)
                else:
                    result[i] = tt_fast_matrix_vec_mul(self._data[i, j], _tt_get_block(j, x_cores), eps)
            if (i, j) in self._aliases:
                k, t = self._aliases[i, j]
                if k in result.keys():
                    result[k] = tt_rank_reduce(
                        tt_add(result.get_row(k), tt_fast_matrix_vec_mul(self._data[i, j], _tt_get_block(t, x_cores), eps)),
                        eps)
                else:
                    result[i] = tt_fast_matrix_vec_mul(self._data[i, j], _tt_get_block(j, x_cores), eps)
        return result



class TTBlockSubMatrix(TTBlockMatrix):
    def __init__(self, data, aliases, transposes):
        self._data = data
        self._aliases = aliases
        self._transposes = transposes


class TTBlockMatrixView:
    """View object that lets you access [row, col] for fixed list index."""
    def __init__(self, data, aliases, transposes, list_index):
        self._data = data
        self._aliases = aliases
        self._transposes = transposes
        self._idx = list_index

    def __getitem__(self, key):
        if not isinstance(key, tuple) or len(key) != 2:
            raise KeyError("Key must be (row, col)")
        return self._data[key][self._idx]

    def items(self):
        """Iterate over ((row, col), value) for fixed index."""
        for coord, values in self._data.items():
            if len(values) > self._idx:
                yield coord, values[self._idx]

    def __repr__(self):
        return f"IndexView({self._idx}) → {{...}}"

    def __iter__(self):
        return iter(self._data)

    def block_local_product(self, XAX_k, XAX_kp1, x_core):
        result = np.zeros_like(x_core, dtype=np.float64)
        for (i, j) in self._data.keys():
            result[:, i] += cached_einsum('lsr,smnS,LSR,rnR->lmL', XAX_k[i, j], self._data[i, j][self._idx], XAX_kp1[i, j], x_core[:, j])
            if (i, j) in self._transposes:
                k, t = self._transposes[i, j]
                result[:, k] += cached_einsum('lsr,smnS,LSR,lmL->rnR', XAX_k[i, j], self._data[i, j][self._idx], XAX_kp1[i, j], x_core[:, t])
            if (i, j) in self._aliases:
                k, t = self._aliases[i,  j]
                result[:, k] += cached_einsum('lsr,smnS,LSR,rnR->lmL', XAX_k[i, j], self._data[i, j][self._idx], XAX_kp1[i, j], x_core[:, t])
        return result

    def keys(self):
        return self._data.keys()

    def tkeys(self):
        return self._data.keys() | self._transposes.values()

    def akeys(self):
        return self._data.keys() | self._aliases.values()

    def all_keys(self):
        return self._data.keys() | self._aliases.values() | self._transposes.values()

def compute_phi_bck_A(Phi_now, core_left, core_A, core_right):
    return cached_einsum('LSR,lML,sMNS,rNR->lsr', Phi_now, core_left, core_A, core_right)


def compute_phi_fwd_A(Phi_now, core_left, core_A, core_right):
    return cached_einsum('lsr,lML,sMNS,rNR->LSR',Phi_now, core_left, core_A, core_right)


def compute_phi_bck_rhs(Phi_now, core_b, core):
    return cached_einsum('BR,bnB,rnR->br', Phi_now, core_b, core)


def compute_phi_fwd_rhs(Phi_now, core_rhs, core):
    return cached_einsum('br,bnB,rnR->BR', Phi_now, core_rhs, core)


def _add_kick_rank(u, v, r_add=2):
    old_r = u.shape[-1]
    uk = np.random.randn(u.shape[0], r_add)  # rx_k x N_k x rz_k+1
    u, Rmat = scp.linalg.qr(np.concatenate((u, uk), 1), check_finite=False, mode="economic", overwrite_a=True)
    v = Rmat[:, :old_r] @ v
    return u, v, u.shape[-1]


def _bck_sweep(
        local_solver,
        x_cores,
        XAX,
        block_A,
        Xb,
        block_b,
        rx,
        N,
        block_size,
        real_tol,
        d,
        swp,
        size_limit,
        eps,
        r_max,
        termination_tol
):
    local_res = np.inf if swp == 0 else 0
    r_max_record = 0
    for k in range(d - 1, -1, -1):
        block_A_k = block_A[k]
        # TODO: This is wrong, shiieet
        if swp > 0:
            previous_solution = x_cores[k]
            solution_now, block_res_old, block_res_new, rhs = local_solver(XAX[k], block_A_k, XAX[k + 1],
                                                                                     Xb[k], block_b[k], Xb[k + 1],
                                                                                     previous_solution,
                                                                                     size_limit, termination_tol)
            local_res = max(local_res, block_res_old)

            solution_now = np.reshape(solution_now, (rx[k] * block_size, N[k] * rx[k + 1])).T
        else:
            solution_now = np.reshape(x_cores[k], (rx[k] * block_size, N[k] * rx[k + 1])).T


        if k > 0:
            if min(rx[k] * block_size, N[k] * rx[k + 1]) > 2*r_max:
                u, s, v = scp.sparse.linalg.svds(solution_now, k=r_max, tol=eps, which="LM")
                idx = np.argsort(s)[::-1]  # descending order
                s = s[idx]
                u = u[:, idx]
                v = v[idx, :]
            else:
                u, s, v = scp.linalg.svd(solution_now, full_matrices=False, check_finite=False, overwrite_a=True, lapack_driver="gesvd")
            v = s.reshape(-1, 1) * v

            r = min(prune_singular_vals(s, real_tol), r_max)
            r_max_record += np.sum(s[r:])
            u = np.reshape(u[:, :r].T, (r, N[k], rx[k + 1]))
            v = v[:r].T.reshape(rx[k], block_size, r)

            x_cores[k] = u
            x_cores[k - 1] = einsum('rdc,cbR->rbdR', x_cores[k - 1], v, optimize=[(0, 1)])
            rx[k] = r

            XAX[k] = {(i, j): compute_phi_bck_A(XAX[k + 1][(i, j)], x_cores[k], block_A_k[(i, j)], x_cores[k]) for (i, j) in block_A_k}

            Xb[k] = {i: compute_phi_bck_rhs(Xb[k + 1][i], block_b[k][i], x_cores[k]) for i in block_b[k]}

        else:
            x_cores[k] = np.reshape(solution_now.T, (rx[k], block_size, N[k], rx[k + 1]))

    return x_cores, XAX, Xb, rx, local_res, r_max_record


def _fwd_sweep(
        local_solver,
        x_cores,
        XAX,
        block_A,
        Xb,
        block_b,
        rx,
        N,
        block_size,
        real_tol,
        d,
        swp,
        size_limit,
        eps,
        r_max,
        termination_tol
):
    local_res = np.inf if swp == 0 else 0
    r_max_record = 0
    for k in range(d):
        block_A_k = block_A[k]
        if swp > 0:
            previous_solution = x_cores[k]
            solution_now, block_res_old, block_res_new, rhs = local_solver(
                XAX[k], block_A_k, XAX[k + 1], Xb[k],
                block_b[k], Xb[k + 1],
                previous_solution, size_limit,
                termination_tol
            )
            local_res = max(local_res, block_res_old)

            solution_now = np.transpose(solution_now, (0, 2, 1, 3))
            solution_now = np.reshape(solution_now, (rx[k] * N[k], block_size * rx[k + 1]))
        else:
            solution_now = np.reshape(x_cores[k], (rx[k] * N[k],  block_size * rx[k + 1]))

        if k < d - 1:
            if min(rx[k] * N[k],  block_size * rx[k + 1]) > 2*r_max:
                u, s, v = scp.sparse.linalg.svds(solution_now, k=r_max, tol=eps, which="LM")
                idx = np.argsort(s)[::-1]  # descending order
                s = s[idx]
                u = u[:, idx]
                v = v[idx, :]
            else:
                u, s, v = scp.linalg.svd(solution_now, full_matrices=False, check_finite=False, overwrite_a=True, lapack_driver="gesvd")

            r = min(prune_singular_vals(s, real_tol), r_max)
            r_max_record += np.sum(s[r:])
            v = s.reshape(-1, 1) * v

            u = u[:, :r].reshape(rx[k], N[k], r)
            v = v[:r].reshape(r, block_size, rx[k + 1])

            x_cores[k] = u
            x_cores[k + 1] = einsum("rbR, Rdk -> rbdk", v, x_cores[k + 1], optimize=[(0, 1)])
            rx[k + 1] = r

            XAX[k + 1] = {(i, j): compute_phi_fwd_A(XAX[k][(i, j)], x_cores[k], block_A_k[(i, j)], x_cores[k]) for
                          (i, j) in block_A_k}
            Xb[k + 1] = {i: compute_phi_fwd_rhs(Xb[k][i], block_b[k][i], x_cores[k]) for i in block_b[k]}

        else:
            x_cores[k] = np.reshape(solution_now, (rx[k], N[k], block_size, rx[k + 1])).transpose(0, 2, 1, 3)

    return x_cores, XAX, Xb, rx, local_res, r_max_record


def _tt_block_als(
        block_A,
        block_b,
        tol,
        termination_tol=1e-3,
        eps=1e-12,
        nswp=22,
        r_max_final=100,
        x0=None,
        local_solver=None,
        refinement=False,
        verbose=False
):
    if verbose:
        print("\n\t---Starting Block-ALS---")
    block_size = np.max(list(k[0] for k in block_A.keys())) + 1
    model_entry = next(iter(block_b.values()))
    x_shape = model_entry[0].shape[1:-1]

    # scale residuals
    rescale = max(block_b.norm, 1e-6)
    block_b.scale(rescale)

    if local_solver is None:
        local_solver = _default_local_solver

    direction = 1
    if x0 is None:
        x_cores = tt_normalise([np.random.randn(1, *c.shape[1:-1], 1) for c in model_entry[:-1]]) + [np.random.randn(1, block_size, *x_shape, 1)]
    else:
        if len(x0[0].shape) > len(x0[-1].shape):
            direction *= -1
        x_cores = x0

    if verbose:
        t0 = time.time()

    N = [c.shape[-2] for c in x_cores]
    d = len(N)

    XAX =  [{key: np.ones((1, 1, 1)) for key in block_A}] + [{key: None for key in block_A} for _ in range(d-1)] + [{key: np.ones((1, 1, 1)) for key in block_A}]  # size is rk x Rk x rk
    Xb = [{key: np.ones((1, 1)) for key in block_b}] + [{key: None for key in block_b} for _ in range(d-1)] + [{key: np.ones((1, 1)) for key in block_b}]   # size is rk x rbk

    r_max_warm_up = min(24, r_max_final)
    size_limit = 0
    if not refinement:
        x_cores = tt_rank_retraction(x_cores, [r_max_warm_up // 2 + 2]*(d-1)) if x0 is not None else x_cores
        size_limit = (r_max_warm_up+1)**2*N[0]/(0.5*np.floor(np.sqrt(d)*d))

    rx = np.array([1] + tt_ranks(x_cores) + [1])
    local_res_fwd = np.inf
    local_res_bwd = np.inf
    trunc_tol = tol/np.sqrt(d)
    refinement = refinement or size_limit == 0

    for swp in range(nswp):
        r_max = r_max_final if swp > 2 or refinement else r_max_warm_up
        x_cores, XAX, Xb, rx, local_res_bwd, rmax_record = _bck_sweep(
            local_solver,
            x_cores,
            XAX,
            block_A,
            Xb,
            block_b,
            rx,
            N,
            block_size,
            trunc_tol,
            d,
            swp,
            size_limit,
            eps,
            r_max,
            termination_tol
        )
        x_cores, XAX, Xb, rx, local_res_fwd, rmax_record = _fwd_sweep(
            local_solver,
            x_cores,
            XAX,
            block_A,
            Xb,
            block_b,
            rx,
            N,
            block_size,
            trunc_tol,
            d,
            swp,
            size_limit,
            eps,
            r_max,
            termination_tol
        )
        if min(local_res_fwd, local_res_bwd) < termination_tol:
            break

        if verbose:
            print('\tStarting Sweep: %d' % swp)
            print(f"\tTrunc loss: {rmax_record}")
            print(f'\tResidual {local_res_fwd}')
            print(f"\tTT-sol rank: {tt_ranks(x_cores)}", flush=True)


    if verbose:
        print("\n\t---Results---")
        print('\tSolution rank is', rx[1:-1])
        print('\tResidual ', min(local_res_fwd, local_res_bwd))
        print('\tNumber of sweeps', swp+1)
        print('\tTime: ', time.time() - t0)
        print('\tTime per sweep: ', (time.time() - t0) / (swp+1), flush=True)

    block_b.scale(1/rescale)

    return tt_scale(rescale, x_cores), min(local_res_fwd, local_res_bwd)

def _default_local_solver(XAX_k, block_A_k, XAX_k1, Xb_k, block_b_k, Xb_k1, previous_solution, _, termination_tol):
    x_shape = previous_solution.shape
    block_size = x_shape[1]
    m = x_shape[0]*x_shape[2]*x_shape[3]
    rhs = np.zeros_like(previous_solution)
    x_shape = (x_shape[1], x_shape[0], x_shape[2], x_shape[3])
    for i in block_b_k:
        rhs[:, i] = cached_einsum('br,bmB,BR->rmR', Xb_k[i], block_b_k[i], Xb_k1[i])
    block_res_old = np.linalg.norm(
        block_A_k.block_local_product(XAX_k, XAX_k1, previous_solution) - rhs)
    def mat_vec(x_vec):
        return np.transpose(block_A_k.block_local_product(
            XAX_k, XAX_k1,
            np.transpose(x_vec.reshape(*x_shape), (1, 0, 2, 3))
        ), (1, 0, 2, 3)).reshape(-1, 1)

    linear_op = scp.sparse.linalg.LinearOperator((block_size * m, block_size * m), matvec=mat_vec)
    max_iter = min(max(2 * int(np.ceil(block_res_old / termination_tol)), 2), 100)
    solution_now, info = scp.sparse.linalg.lgmres(linear_op, np.transpose(
        rhs - block_A_k.block_local_product(XAX_k, XAX_k1, previous_solution), (1, 0, 2, 3)).reshape(-1, 1), rtol=1e-3*block_res_old, maxiter=max_iter)
    solution_now = np.transpose(solution_now.reshape(*x_shape), (1, 0, 2, 3))

    solution_now += previous_solution
    block_res_new = np.linalg.norm(
        block_A_k.block_local_product(XAX_k, XAX_k1, solution_now) - rhs)

    if block_res_old < block_res_new:
        solution_now = previous_solution

    return solution_now, block_res_old, min(block_res_old, block_res_new), rhs


def tt_rl_orthogonalise_py(train_tt: List[np.array]):
    dim = len(train_tt)
    if dim == 1:
        return train_tt
    for idx in range(dim - 1, 0, -1):
        shape_p1 = train_tt[idx].shape
        shape = train_tt[idx - 1].shape
        Q_T, R = np.linalg.qr(train_tt[idx].reshape(shape_p1[0], -1).T)
        train_tt[idx] = Q_T.T.reshape(-1, *shape_p1[1:-1], shape_p1[-1])
        train_tt[idx - 1] = (train_tt[idx - 1].reshape(-1, R.shape[-1]) @ R.T).reshape(-1, *shape[1:-1],
                                                                                       train_tt[idx].shape[0])
    return train_tt

def tt_rank_reduce_py(train_tt: List[np.array], eps=1e-18):
    """ Might reduce TT-rank """
    dim = len(train_tt)
    ranks = np.array([1] + tt_ranks(train_tt) + [1])
    if dim == 1 or np.all(ranks==1):
        return train_tt
    eps = eps / np.sqrt(dim - 1)
    train_tt = tt_rl_orthogonalise_py(train_tt)
    rank = 1
    for idx, tt_core in enumerate(train_tt[:-1]):
        idx_shape = tt_core.shape
        next_idx_shape = train_tt[idx + 1].shape
        k = len(idx_shape) - 1
        u, s, v_t = scp.linalg.svd(train_tt[idx].reshape(rank * np.prod(idx_shape[1:k], dtype=int), -1), full_matrices=False, check_finite=False, overwrite_a=True)
        next_rank = prune_singular_vals(s, eps)
        s = s[:next_rank]
        u = u[:, :next_rank]
        v_t = v_t[:next_rank, :]
        train_tt[idx] = u.reshape(rank, *idx_shape[1:-1], next_rank)
        train_tt[idx + 1] = (
            s.reshape(-1, 1) * v_t @ train_tt[idx + 1].reshape(v_t.shape[-1], -1)
        ).reshape(next_rank, *next_idx_shape[1:-1], -1)
        rank = next_rank
    return train_tt



def tt_restarted_block_als(
        block_A,
        block_b,
        rank_restriction,
        tol,
        termination_tol=1e-3,
        eps=1e-12,
        num_restarts=3,
        inner_m=10,
        x0=None,
        local_solver=None,
        refinement=False,
        verbose=False
):
    rhs = block_b
    x_cores, res = _tt_block_als(block_A, rhs, tol, termination_tol, eps, inner_m, rank_restriction, x0, local_solver, refinement, verbose)
    if res < termination_tol:
        if verbose:
            print(f"\n\tTerminated on local criterion,  Error<{termination_tol}")
        return x_cores, res
    Ax = block_A.matvec(x_cores)
    rhs = rhs - Ax
    rhs_norm = rhs.norm
    if rhs_norm < termination_tol:
        if verbose:
            print(f"\n\tTerminated on global criterion,  Error={rhs_norm}")
        return x_cores, res
    prev_rhs_norm = rhs_norm
    if verbose:
        print(f"\n\tGlobal Error={rhs_norm}")
    for i in range(1, num_restarts):
        if verbose:
            print(f"\n\t---Restart {i}")
        new_x_cores, res = _tt_block_als(block_A, rhs, tol, termination_tol, eps, inner_m, rank_restriction, None, local_solver, refinement, verbose)
        if res < termination_tol:
            if verbose:
                print(f"\n\tTerminated on local criterion,  Error<{termination_tol}")
            x_cores = tt_rank_reduce_py(tt_add(x_cores, new_x_cores), eps=eps)
            break
        Ax = block_A.matvec(new_x_cores)
        rhs = rhs - Ax
        rhs_norm = rhs.norm
        if rhs_norm > prev_rhs_norm:
            if verbose:
                print(f"\n\tTerminated on instability ({rhs_norm} > {prev_rhs_norm})!")
            break
        elif rhs_norm < termination_tol:
            if verbose:
                print(f"\n\tTerminated on global criterion,  Error={rhs_norm}")
            x_cores = tt_rank_reduce_py(tt_add(x_cores, new_x_cores), eps=eps)
            break
        if verbose:
            print(f"\n\tGlobal Error={rhs_norm}")
        prev_rhs_norm = rhs_norm
        x_cores = tt_rank_reduce_py(tt_add(x_cores, new_x_cores), eps=eps)
    else:
        if verbose:
            print(f"\n\tNumber of restarts exhausted,  Error={rhs_norm}")

    return x_cores, res




