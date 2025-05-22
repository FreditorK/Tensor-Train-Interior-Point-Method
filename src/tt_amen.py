import sys
import os
import time
import copy

import numpy as np

sys.path.append(os.getcwd() + '/../')

from src.tt_ops import *
from opt_einsum import contract as einsum
from sklearn.utils.extmath import randomized_svd

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

    def block_local_product(self, Xb_k, Xb_kp1, nrmsc, shape):
        result = np.zeros(shape)
        for i in self._data.keys():
            result[:, i] += cached_einsum('br,bnB,BR->rnR', Xb_k[i], nrmsc * self._data[i][self._list_index], Xb_kp1[i])
        return result



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
        return self._data.keys() | self._aliases.values() | self._transposes.values()

    def __iter__(self):
        return iter(self._data)


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
        result = np.zeros_like(x_core)
        for (i, j) in self._data.keys():
            result[:, i] += cached_einsum('lsr,smnS,LSR,rnR->lmL', XAX_k[i, j], self._data[i, j][self._idx], XAX_kp1[i, j], x_core[:, j])
            if (i, j) in self._transposes:
                k, t = self._transposes[i, j]
                result[:, k] += cached_einsum('lsr,smnS,LSR,lmL->rnR', XAX_k[i, j], self._data[i, j][self._idx], XAX_kp1[i, j], x_core[:, t])
            if (i, j) in self._aliases:
                k, t = self._aliases[i,  j]
                result[:, k] += cached_einsum('lsr,smnS,LSR,rnR->lmL', XAX_k[i, j], self._data[i, j][self._idx], XAX_kp1[i, j], x_core[:, t])
        return result

    def compressed_block_local_product(self, ZAX_k, ZAX_kp1, x_core, shape):
        result = np.zeros(shape)
        for (i, j) in self._data.keys():
            result[:, i] += cached_einsum('lsr,smnS,LSR,rnR->lmL', ZAX_k[i, j], self._data[i, j][self._idx], ZAX_kp1[i, j], x_core[:, j])
            if (i, j) in self._transposes:
                k, t = self._transposes[i, j]
                result[:, k] += cached_einsum('lsr,snmS,LSR,rnR->lmL', ZAX_k[k, t], self._data[i, j][self._idx], ZAX_kp1[k, t], x_core[:, t])
            if (i, j) in self._aliases:
                k, t = self._aliases[i,  j]
                result[:, k] += cached_einsum('lsr,smnS,LSR,rnR->lmL', ZAX_k[i, j], self._data[i, j][self._idx], ZAX_kp1[i, j], x_core[:, t])
        return result


    def lcompressed_block_local_product(self, ZAX_k, XAX_kp1, x_core, shape):
        result = np.zeros(shape)
        for (i, j) in self._data.keys():
            result[:, i] += cached_einsum('lsr,smnS,LSR,rnR->lmL', ZAX_k[i, j], self._data[i, j][self._idx], XAX_kp1[i, j], x_core[:, j])
            if (i, j) in self._transposes:
                k, t = self._transposes[i, j]
                result[:, k] += cached_einsum('lsr,snmS,LSR,rnL->lmR', ZAX_k[k, t], self._data[i, j][self._idx], XAX_kp1[i, j], x_core[:, t])
            if (i, j) in self._aliases:
                k, t = self._aliases[i,  j]
                result[:, k] += cached_einsum('lsr,smnS,LSR,rnR->lmL', ZAX_k[i, j], self._data[i, j][self._idx], XAX_kp1[i, j], x_core[:, t])
        return result


    def rcompressed_block_local_product(self, XAX_k, ZAX_kp1, x_core, shape):
        result = np.zeros(shape)
        for (i, j) in self._data.keys():
            result[:, i] += cached_einsum('lsr,smnS,LSR,rnR->lmL', XAX_k[i, j], self._data[i, j][self._idx], ZAX_kp1[i, j], x_core[:, j])
            if (i, j) in self._transposes:
                k, t = self._transposes[i, j]
                result[:, k] += cached_einsum('lsr,snmS,LSR,lnR->rmL', XAX_k[i, j], self._data[i, j][self._idx], ZAX_kp1[k, t], x_core[:, t])
            if (i, j) in self._aliases:
                k, t = self._aliases[i,  j]
                result[:, k] += cached_einsum('lsr,smnS,LSR,rnR->lmL', XAX_k[i, j], self._data[i, j][self._idx], ZAX_kp1[i, j], x_core[:, t])
        return result

    def keys(self):
        return self._data.keys()

    def tkeys(self):
        return self._data.keys() | self._transposes.values()

    def akeys(self):
        return self._data.keys() | self._aliases.values()

    def all_keys(self):
        return self._data.keys() | self._aliases.values() | self._transposes.values()

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
        z_cores,
        normx,
        XAX,
        ZAX,
        block_A,
        normA,
        Xb,
        Zb,
        block_b,
        normb,
        nrmsc,
        rx,
        rz,
        N,
        block_size,
        real_tol,
        d,
        swp,
        size_limit,
        eps,
        r_max,
        kick_rank,
        last
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

            if not last:
                Az = block_A[k].compressed_block_local_product(ZAX[k], ZAX[k + 1], solution_now, shape=(rz[k], block_size, N[k], rz[k + 1]))
                rhsz = block_b[k].block_local_product(Zb[k], Zb[k + 1], nrmsc, (rz[k], block_size, N[k], rz[k + 1]))
                resz = rhsz.__isub__(Az)

            solution_now = np.reshape(solution_now, (rx[k] * block_size, N[k] * rx[k + 1])).T
        else:
            solution_now = np.reshape(x_cores[k], (rx[k] * block_size, N[k] * rx[k + 1])).T
            if not last:
                resz = np.reshape(z_cores[k], (rz[k] * block_size, N[k] * rz[k + 1])).T

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

            if not last:
                uz, sz, vz = randomized_svd(np.reshape(resz, (rz[k] * block_size, N[k] * rz[k + 1])).T,
                                            n_components=kick_rank, n_iter=3)
                uz = uz.T.reshape(kick_rank, N[k], rz[k + 1])
                vz = np.reshape((sz.reshape(-1, 1) * vz).T, (rz[k], block_size, kick_rank))
                z_cores[k] = uz
                z_cores[k - 1] = einsum('rdc,cbR->rbdR', z_cores[k - 1], vz, optimize=[(0, 1)])

            if swp > 0:
                r_start = min(prune_singular_vals(s, real_tol), r_max)
                solution_now = np.reshape((u[:, :r_start] @ v[:r_start]).T, (rx[k], block_size, N[k], rx[k + 1]))
                res = block_A[k].block_local_product(XAX[k], XAX[k + 1], solution_now) - rhs
                r = r_start
                for r in range(r_start - 1, 0, -1):
                    res -= block_A[k].block_local_product(XAX[k], XAX[k + 1],
                                                np.reshape((u[:, None, r] @ v[None, r, :]).T,
                                                           (rx[k], block_size, N[k], rx[k + 1])))
                    if np.linalg.norm(res) / norm_rhs > max(2 * real_tol, block_res_new):
                        break
                r += 1
                u = np.reshape(u[:, :r].T, (r, N[k], rx[k + 1]))
                v = v[:r].T.reshape(rx[k], block_size, r)
                if not last:
                    # amen enhancement
                    Axz = block_A[k].lcompressed_block_local_product(ZAX[k], XAX[k + 1], solution_now, shape=(rz[k], block_size, N[k], rx[k + 1]))
                    rhsxz = block_b[k].block_local_product(Zb[k], Xb[k + 1], nrmsc, (rz[k], block_size, N[k], rx[k + 1]))
                    resxz = rhsxz.__isub__(Axz)
                    uz, _, _ = randomized_svd(np.reshape(resxz, (rz[k] * block_size, N[k] * rx[k + 1])).T, n_components=kick_rank, n_iter=3)
                    uz = uz.T.reshape(kick_rank, N[k], rx[k + 1])
                    u = np.concatenate((np.reshape(u, (r, N[k], rx[k + 1])), uz), axis=0)
                    u, R = scip.linalg.qr(u.reshape(-1, N[k]*rx[k+1]).T, mode="economic", check_finite=False, overwrite_a=True)
                    u = u.T.reshape(-1, N[k], rx[k+1])
                    v = np.concatenate((v, np.zeros((rx[k], block_size, kick_rank))), axis=-1)
                    v = einsum("Rdk, kr -> Rdr", v, R.T, optimize=[(0, 1)])
                    r = u.shape[0]

                nrmsc *= (normA[k - 1] * normx[k - 1]) / normb[k - 1]
            else:
                r = min(prune_singular_vals(s, real_tol), r_max)
                u = np.reshape(u[:, :r].T, (r, N[k], rx[k + 1]))
                v = v[:r].T.reshape(rx[k], block_size, r)
            x_cores[k] = u
            x_cores[k - 1] = einsum('rdc,cbR->rbdR', x_cores[k - 1], v, optimize=[(0, 1)])
            norm_now = np.linalg.norm(x_cores[k - 1])
            normx[k - 1] *= norm_now
            x_cores[k - 1] /= norm_now
            rx[k] = r

            XAX[k] = {(i, j): _compute_phi_bck_A(XAX[k + 1][(i, j)], x_cores[k], block_A[k][(i, j)], x_cores[k]) for (i, j) in block_A[k]}
            normA[k - 1] = max(
                np.sqrt(sum((1 + ((i, j) in block_A[k]._aliases) + ((i, j) in block_A[k]._transposes))*np.sum(XAX[k][i, j] ** 2) for (i, j) in block_A[k]._data)),
                eps
            )
            XAX[k] = {(i, j): XAX[k][(i, j)] / normA[k - 1] for (i, j) in block_A[k]}

            Xb[k] = {i: _compute_phi_bck_rhs(Xb[k + 1][i], block_b[k][i], x_cores[k]) for i in block_b[k]}
            normb[k - 1] = max(np.sqrt(sum(np.sum(v ** 2) for v in Xb[k].values())), eps)

            Xb[k] = {i: Xb[k][i] / normb[k - 1] for i in block_b[k]}

            if not last:
                ZAX[k] = {(i, j): _compute_phi_bck_A(ZAX[k + 1][(i, j)], z_cores[k], block_A[k][(i, j)], x_cores[k]) for (i, j) in block_A[k]}
                ZAX[k].update({(l, t): _compute_phi_bck_A(ZAX[k + 1][(l, t)], z_cores[k], np.transpose(block_A[k][(i, j)], (0, 2, 1, 3)), x_cores[k]) for (i, j), (l, t) in block_A[k]._transposes.items()})
                ZAX[k] = {(i, j): ZAX[k][(i, j)] / normA[k - 1] for (i, j) in block_A[k].tkeys()}
                Zb[k] = {i: _compute_phi_bck_rhs(Zb[k + 1][i], block_b[k][i], z_cores[k]) for i in block_b[k]}
                Zb[k] = {i: Zb[k][i] / normb[k - 1] for i in block_b[k]}

            nrmsc *= normb[k - 1] / (normA[k - 1] * normx[k - 1])
        else:
            x_cores[k] = np.reshape(solution_now.T, (rx[k], block_size, N[k], rx[k + 1]))
            if not last:
                z_cores[k] = np.reshape(resz, (rz[k], block_size, N[k], rz[k + 1]))

    return x_cores, normx, XAX, normA, Xb, normb, nrmsc, rx, local_res, local_dx


def _fwd_sweep(
        local_solver,
        x_cores,
        z_cores,
        normx,
        XAX,
        ZAX,
        block_A,
        normA,
        Xb,
        Zb,
        block_b,
        normb,
        nrmsc,
        rx,
        rz,
        N,
        block_size,
        real_tol,
        d,
        swp,
        size_limit,
        eps,
        r_max,
        kick_rank,
        last
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

            if not last:
                Az = block_A[k].compressed_block_local_product(ZAX[k], ZAX[k + 1], solution_now, shape=(rz[k], block_size, N[k], rz[k + 1]))
                rhsz = block_b[k].block_local_product(Zb[k], Zb[k + 1], nrmsc, (rz[k], block_size, N[k], rz[k + 1]))
                resz = np.transpose(rhsz.__isub__(Az), (0, 2, 1, 3)).reshape(rz[k] * N[k], block_size * rz[k + 1])

            solution_now = np.transpose(solution_now, (0, 2, 1, 3))
            solution_now = np.reshape(solution_now, (rx[k] * N[k], block_size * rx[k + 1]))
        else:
            solution_now = np.reshape(x_cores[k], (rx[k] * N[k],  block_size * rx[k + 1]))
            if not last:
                resz = np.reshape(z_cores[k], (rz[k] * N[k], block_size * rz[k + 1]))

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

            if not last:
                uz, sz, vz = randomized_svd(resz, n_components=kick_rank, n_iter=3)
                uz = uz[:, :kick_rank]
                sz = sz[:kick_rank]
                vz = vz[:kick_rank]
                uz = np.reshape(uz, (rz[k], N[k], kick_rank))
                vz = np.reshape(sz.reshape(-1, 1) * vz, (kick_rank, block_size, rz[k + 1]))
                z_cores[k] = uz
                z_cores[k + 1] = einsum("rbR, Rdk -> rbdk", vz, z_cores[k + 1], optimize=[(0, 1)])

            if swp > 0:
                r_start = min(prune_singular_vals(s, real_tol), r_max)
                solution_now = einsum("rbR, Rdk -> rbdk", u[:, :, :r_start], v[:r_start], optimize=[(0, 1)])
                res = block_A[k].block_local_product(XAX[k], XAX[k + 1], np.transpose(solution_now, (0, 2, 1, 3))) - rhs
                r = r_start
                for r in range(r_start - 1, 0, -1):
                    res -= block_A[k].block_local_product(XAX[k], XAX[k + 1], einsum("rbR, Rdk -> rdbk", u[:, :, None, r], v[None, r], optimize=[(0, 1)]))
                    if np.linalg.norm(res) / norm_rhs > max(2 * real_tol, block_res_new):
                        break
                r += 1
                if not last:
                    # amen enhancement
                    Axz = block_A[k].rcompressed_block_local_product(XAX[k], ZAX[k + 1], einsum("rbR, Rdk -> rdbk", u[:, :, :r], v[:r], optimize=[(0, 1)]), shape=(rx[k], block_size, N[k], rz[k + 1]))
                    rhsxz = block_b[k].block_local_product(Xb[k], Zb[k + 1], nrmsc, (rx[k], block_size, N[k], rz[k + 1]))
                    resxz = np.transpose(rhsxz.__isub__(Axz), (0, 2, 1, 3))
                    uz, _, _ = randomized_svd(np.reshape(resxz, (rx[k] * N[k], block_size * rz[k + 1])), n_components=kick_rank, n_iter=3)
                    uz = np.reshape(uz, (rx[k], N[k], kick_rank))
                    u = np.concatenate((u[:, :, :r], uz), axis=-1)
                    u, R = scip.linalg.qr(u.reshape(rx[k]*N[k], -1), mode="economic", check_finite=False, overwrite_a=True)
                    u = u.reshape(rx[k], N[k], -1)
                    v = np.concatenate((v[:r], np.zeros((kick_rank, block_size, rx[k + 1]))), axis=0)
                    v = einsum("rR, Rdk -> rdk", R, v, optimize=[(0, 1)])
                    r = v.shape[0]
                else:
                    u = u[:, :, :r]
                    v = v[:r]
                nrmsc *= normA[k] * normx[k] / normb[k]
            else:
                r = min(prune_singular_vals(s, real_tol), r_max)
                u = u[:, :, :r]
                v = v[:r]

            v = einsum("rbR, Rdk -> rbdk", v, x_cores[k + 1], optimize=[(0, 1)])
            norm_now = np.linalg.norm(v)
            normx[k] *= norm_now
            x_cores[k] = u
            x_cores[k + 1] = (v / norm_now).reshape(r, block_size, N[k + 1], rx[k + 2])
            rx[k + 1] = r

            XAX[k + 1] = {(i, j): _compute_phi_fwd_A(XAX[k][(i, j)], x_cores[k], block_A[k][(i, j)], x_cores[k]) for
                          (i, j) in block_A[k]}
            normA[k] = max(
                np.sqrt(
                    sum((1 + ((i, j) in block_A[k]._aliases) + ((i, j) in block_A[k]._transposes))*np.sum(XAX[k+1][i, j] ** 2) for (i, j) in block_A[k]._data)),
                eps
            )
            XAX[k + 1] = {(i, j): XAX[k + 1][(i, j)] / normA[k] for (i, j) in block_A[k]}
            Xb[k + 1] = {i: _compute_phi_fwd_rhs(Xb[k][i], block_b[k][i], x_cores[k]) for i in block_b[k]}
            normb[k] = max(np.sqrt(sum(np.sum(v ** 2) for v in Xb[k + 1].values())), eps)
            Xb[k + 1] = {i: Xb[k + 1][i] / normb[k] for i in block_b[k]}

            if not last:
                ZAX[k + 1] = {(i, j): _compute_phi_fwd_A(ZAX[k][(i, j)], z_cores[k], block_A[k][(i, j)], x_cores[k]) for (i, j) in block_A[k]}
                ZAX[k + 1].update({(l, t): _compute_phi_fwd_A(ZAX[k][(l, t)], z_cores[k], np.transpose(block_A[k][(i, j)], (0, 2, 1, 3)), x_cores[k]) for (i, j), (l, t) in block_A[k]._transposes.items()})
                ZAX[k + 1] = {(i, j): ZAX[k + 1][(i, j)] / normA[k] for (i, j) in block_A[k].tkeys()}
                Zb[k + 1] = {i: _compute_phi_fwd_rhs(Zb[k][i], block_b[k][i], z_cores[k]) for i in block_b[k]}
                Zb[k + 1] = {i: Zb[k + 1][i] / normb[k] for i in block_b[k]}

            nrmsc *= normb[k] / (normA[k] * normx[k])

        else:
            x_cores[k] = np.reshape(solution_now, (rx[k], N[k], block_size, rx[k + 1])).transpose(0, 2, 1, 3)
            if not last:
                z_cores[k] = np.reshape(resz, (rz[k], N[k], block_size, rz[k + 1])).transpose(0, 2, 1, 3)

    return x_cores, normx, XAX, normA, Xb, normb, nrmsc, rx, local_res, local_dx


def tt_block_amen(block_A, block_b, tol, eps=1e-10, nswp=22, x0=None, size_limit=None, local_solver=None, kick_rank=2, amen=False, verbose=False):

    block_size = np.max(list(k[0] for k in block_A.keys())) + 1
    model_entry = next(iter(block_b.values()))
    x_shape = model_entry[0].shape[1:-1]

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


    normA = np.ones(d - 1) # norm of each row in the block matrix
    normb = np.ones(d - 1) # norm of each row of the rhs
    nrmsc = 1.0
    normx = np.ones((d - 1))
    real_tol = (tol / np.sqrt(d))
    r_max_final = block_size*int(np.ceil(np.sqrt(d)*d)) + block_size*int(np.ceil(np.sqrt(block_size)))
    size_limit = (r_max_final)**2*N[0]/(np.sqrt(d)*d) if size_limit is None else size_limit
    r_max_part0 = min(15, r_max_final - 1)
    r_max_part = np.linspace(r_max_part0, r_max_final, num=nswp, dtype=int)
    x_cores = tt_rank_retraction(x_cores, [r_max_part0]*(d-1)) if x0 is not None else x_cores
    rx = np.array([1] + tt_ranks(x_cores) + [1])

    ZAX = None
    Zb = None
    z_cores = None
    rz = None
    if amen:
        ZAX = [{key: np.ones((1, 1, 1)) for key in block_A.tkeys()}] + [{key: None for key in block_A.tkeys()} for _ in range(d - 1)] + [
            {key: np.ones((1, 1, 1)) for key in block_A.tkeys()}]  # size is rk x Rk x rk
        Zb = [{key: np.ones((1, 1)) for key in block_b}] + [{key: None for key in block_b} for _ in range(d - 1)] + [
            {key: np.ones((1, 1)) for key in block_b}]  # size is rk x rbk
        z_cores = (
                [np.divide(1, np.prod(x_cores[0].shape[1:-1]) * kick_rank**2)*np.random.randn(*x_cores[0].shape[:-1], kick_rank)]
                + [np.divide(1, np.prod(c.shape[1:-1]) * kick_rank**2) * np.random.randn(kick_rank, *c.shape[1:-1], kick_rank) for c in x_cores[1:-1]]
                + [np.divide(1, np.prod(x_cores[-1].shape[1:-1]) * kick_rank**2) * np.random.randn(kick_rank, *x_cores[-1].shape[1:])]
        )
        rz = np.array([1] + tt_ranks(z_cores) + [1])
    last = False

    for swp in range(nswp):
        r_max = r_max_part[swp]
        if direction > 0:
            x_cores, normx, XAX, normA, Xb, normb, nrmsc, rx, local_res, local_dx = _bck_sweep(
                local_solver,
                x_cores,
                z_cores,
                normx,
                XAX,
                ZAX,
                block_A,
                normA,
                Xb,
                Zb,
                block_b,
                normb,
                nrmsc,
                rx,
                rz,
                N,
                block_size,
                real_tol,
                d,
                swp,
                size_limit,
                eps,
                r_max,
                kick_rank,
                last or not amen
            )
        else:
            x_cores, normx, XAX, normA, Xb, normb, nrmsc, rx, local_res, local_dx = _fwd_sweep(
                local_solver,
                x_cores,
                z_cores,
                normx,
                XAX,
                ZAX,
                block_A,
                normA,
                Xb,
                Zb,
                block_b,
                normb,
                nrmsc,
                rx,
                rz,
                N,
                block_size,
                real_tol,
                d,
                swp,
                size_limit,
                eps,
                r_max,
                kick_rank,
                last or not amen
            )

        if verbose:
            print('\tStarting Sweep:\n\tMax num of sweeps: %d' % swp)
            print(f'\tDirection {direction}')
            print(f'\tResidual {local_res}')
            print(f"\tTT-sol rank: {tt_ranks(x_cores)}", flush=True)

        if last:
            break
        if local_res < tol or local_dx < eps:
            last = True

        direction *= -1


    if verbose:
        print("\n\t---Results---")
        print('\tSolution rank is', rx[1:-1])
        print('\tResidual ', local_res)
        print('\tNumber of sweeps', swp+1)
        print('\tTime: ', time.time() - t0)
        print('\tTime per sweep: ', (time.time() - t0) / (swp+1), flush=True)

    normx = np.exp(np.sum(np.log(normx)) / d)

    return [normx * core for core in x_cores], np.max(local_res)



def tt_divide(vec_tt_1, vec_tt_2, degenerate=False, eps=1e-10):
    b = vec_tt_1
    A = tt_diag(vec_tt_2)
    if degenerate:
        A = tt_add(A, tt_scale(0.5*eps, tt_identity(len(A))))
    A = tt_rank_reduce(A, eps)
    sol, _ = tt_amen(A, b)
    return sol

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
            return np.transpose(block_A_k.block_local_product(
                XAX_k, XAX_k1,
                np.transpose(x_vec.reshape(*x_shape), (1, 0, 2, 3))
            ), (1, 0, 2, 3)).reshape(-1, 1)

        linear_op = scip.sparse.linalg.LinearOperator((block_size * m, block_size * m), matvec=mat_vec)
        solution_now, info = scip.sparse.linalg.bicgstab(linear_op, np.transpose(
            rhs - block_A_k.block_local_product(XAX_k, XAX_k1, previous_solution), (1, 0, 2, 3)).reshape(-1, 1), rtol=rtol)
        solution_now = np.transpose(solution_now.reshape(*x_shape), (1, 0, 2, 3))

    solution_now += previous_solution
    block_res_new = np.linalg.norm(
        block_A_k.block_local_product(XAX_k, XAX_k1, solution_now) - rhs) / norm_rhs
    block_res_old = np.linalg.norm(
        block_A_k.block_local_product(XAX_k, XAX_k1, previous_solution) - rhs) / norm_rhs

    if block_res_old < block_res_new:
        solution_now = previous_solution

    return solution_now, block_res_old, min(block_res_old, block_res_new), rhs, norm_rhs





