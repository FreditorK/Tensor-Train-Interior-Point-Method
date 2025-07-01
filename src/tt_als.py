import sys
import os
import time

import numpy as np

sys.path.append(os.getcwd() + '/../')

from src.tt_ops import *
from opt_einsum import contract as einsum
from sksparse.cholmod import cholesky as sparse_cholesky

def _tt_get_block(i, block_matrix_tt):
    b = np.argmax([len(c.shape) for c in block_matrix_tt])
    return block_matrix_tt[:b] + [block_matrix_tt[b][:, i]] + block_matrix_tt[b+1:]

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

    def block_local_product(self, Xb_k, Xb_kp1, nrmsc, shape):
        result = np.zeros(shape, dtype=np.float64)
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

    def block_product(self, x_cores, op_tol, eps=1e-12):
        result = TTBlockVector()
        for (i, j) in self._data.keys():
            if i in result.keys():
                result[i] = tt_rank_reduce(tt_add(result.get_row(i), tt_mat_vec_mul(self._data[i, j], _tt_get_block(j, x_cores), op_tol, eps)), eps)
            else:
                result[i] = tt_mat_vec_mul(self._data[i, j], _tt_get_block(j, x_cores), op_tol, eps)
            if (i, j) in self._transposes:
                k, t = self._transposes[i, j]
                if k in result.keys():
                    result[k] = tt_rank_reduce(
                        tt_add(result.get_row(k), tt_mat_vec_mul(tt_transpose(self._data[i, j]), _tt_get_block(t, x_cores), op_tol, eps)),
                        eps)
                else:
                    result[k] = tt_mat_vec_mul(tt_transpose(self._data[i, j]), _tt_get_block(t, x_cores), op_tol, eps)
            if (i, j) in self._aliases:
                k, t = self._aliases[i, j]
                if k in result.keys():
                    result[k] = tt_rank_reduce(
                        tt_add(result.get_row(k), tt_mat_vec_mul(self._data[i, j], _tt_get_block(t, x_cores), op_tol, eps)),
                        eps)
                else:
                    result[k] = tt_mat_vec_mul(self._data[i, j], _tt_get_block(t, x_cores), op_tol, eps)
        return result
    
    def get_submatrix(self, row_index, col_index):
        submatrix = TTBlockMatrix()
        submatrix._data = {(i, j): v for (i, j), v in self._data.items() if i <= row_index and j <= col_index}
        submatrix ._aliases = {(i, j):  (k, t) for (i, j), (k, t) in self._aliases.items() if k <= row_index and t <= col_index}
        submatrix ._transposes = {(i, j):  (k, t) for (i, j), (k, t) in self._transposes.items() if k <= row_index and t <= col_index}
        return submatrix


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

    def compressed_block_local_product(self, ZAX_k, ZAX_kp1, x_core, shape):
        result = np.zeros(shape, dtype=np.float64)
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
        result = np.zeros(shape, dtype=np.float64)
        for (i, j) in self._data.keys():
            result[:, i] += cached_einsum('lsr,smnS,LSR,rnR->lmL', ZAX_k[i, j], self._data[i, j][self._idx], XAX_kp1[i, j], x_core[:, j])
            if (i, j) in self._transposes:
                k, t = self._transposes[i, j]
                result[:, k] += cached_einsum('lsr,snmS,RSL,rnR->lmL', ZAX_k[k, t], self._data[i, j][self._idx], XAX_kp1[i, j], x_core[:, t])
            if (i, j) in self._aliases:
                k, t = self._aliases[i,  j]
                result[:, k] += cached_einsum('lsr,smnS,LSR,rnR->lmL', ZAX_k[i, j], self._data[i, j][self._idx], XAX_kp1[i, j], x_core[:, t])
        return result


    def rcompressed_block_local_product(self, XAX_k, ZAX_kp1, x_core, shape):
        result = np.zeros(shape, dtype=np.float64)
        for (i, j) in self._data.keys():
            result[:, i] += cached_einsum('lsr,smnS,LSR,rnR->lmL', XAX_k[i, j], self._data[i, j][self._idx], ZAX_kp1[i, j], x_core[:, j])
            if (i, j) in self._transposes:
                k, t = self._transposes[i, j]
                result[:, k] += cached_einsum('rsl,snmS,LSR,rnR->lmL', XAX_k[i, j], self._data[i, j][self._idx], ZAX_kp1[k, t], x_core[:, t])
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

def compute_phi_bck_A(Phi_now, core_left, core_A, core_right):
    return cached_einsum('LSR,lML,sMNS,rNR->lsr', Phi_now, core_left, core_A, core_right)


def compute_phi_fwd_A(Phi_now, core_left, core_A, core_right):
    return cached_einsum('lsr,lML,sMNS,rNR->LSR',Phi_now, core_left, core_A, core_right)


def compute_phi_bck_rhs(Phi_now, core_b, core):
    return cached_einsum('BR,bnB,rnR->br', Phi_now, core_b, core)


def compute_phi_fwd_rhs(Phi_now, core_rhs, core):
    return cached_einsum('br,bnB,rnR->BR', Phi_now, core_rhs, core)



def truncated_svd(matrix, trunc_rank):
    u, s, v = scp.linalg.svd(matrix, full_matrices=False, check_finite=False, overwrite_a=True)
    return u[:, :trunc_rank], s[:trunc_rank].reshape(-1, 1) * v[:trunc_rank]


def _bck_sweep(
        local_solver,
        x_cores,
        z_cores,
        XAX,
        ZAX,
        block_A,
        Xb,
        Zb,
        block_b,
        rx,
        rz,
        N,
        block_size,
        trunc_tol,
        d,
        swp,
        eps,
        r_max,
        kick_rank,
        last,
        amen,
        lgmres_discount,
        direct_solve_failure
):
    local_res = np.inf if swp == 0 else 0
    local_dx = np.inf if swp == 0 else 0
    for k in range(d - 1, -1, -1):
        block_A_k = block_A[k]
        block_b_k = block_b[k]
        if swp > 0 and not last:
            previous_solution = x_cores[k]
            solution_now, block_res_old, block_res_new, rhs, norm_rhs, lgmres_discount, direct_solve_failure  = local_solver(XAX[k], block_A_k, XAX[k + 1],
                                                                                     Xb[k], block_b_k, Xb[k + 1],
                                                                                     previous_solution, r_max**(2/3), lgmres_discount, not direct_solve_failure )

            local_res = max(local_res, block_res_old)
            dx = np.linalg.norm(solution_now - previous_solution) / np.linalg.norm(solution_now)
            local_dx = max(dx, local_dx)

            if amen:
                Az = block_A_k.compressed_block_local_product(ZAX[k], ZAX[k + 1], solution_now, shape=(rz[k], block_size, N[k], rz[k + 1]))
                rhsz = block_b_k.block_local_product(Zb[k], Zb[k + 1], 1, (rz[k], block_size, N[k], rz[k + 1]))
                resz = np.reshape(rhsz.__isub__(Az), (rz[k] * block_size, N[k] * rz[k + 1])).T

            scales = np.array([np.linalg.norm(solution_now[:, k]) for k in range(solution_now.shape[1])]).reshape(1, -1, 1, 1)
            solution_now = np.reshape(scales*solution_now, (rx[k] * block_size, N[k] * rx[k + 1])).T
        else:
            solution_now = x_cores[k]
            scales = np.array([np.linalg.norm(solution_now[:, k]) for k in range(solution_now.shape[1])]).reshape(1, -1, 1, 1)
            solution_now = np.reshape(scales*solution_now, (rx[k] * block_size, N[k] * rx[k + 1])).T
            if amen and not last:
                resz = np.reshape(z_cores[k], (rz[k] * block_size, N[k] * rz[k + 1])).T

        if k > 0:
            if min(rx[k] * block_size, N[k] * rx[k + 1]) > 2*r_max:
                u, s, v = scp.sparse.linalg.svds(solution_now, k=r_max, tol=eps, which="LM")
                idx = np.argsort(s)[::-1]  # descending order
                s = s[idx]
                u = u[:, idx]
                v = v[idx, :]
            else:
                u, s, v = scp.linalg.svd(solution_now, full_matrices=False, check_finite=False, overwrite_a=True)
            v = s.reshape(-1, 1) * v

            if swp > 0 and not last:
                r_start = min(prune_singular_vals(s, eps), r_max)
                solution_now = np.reshape((u[:, :r_start] @ v[:r_start]).T, (rx[k], block_size, N[k], rx[k + 1]))
                res = block_A_k.block_local_product(XAX[k], XAX[k + 1], solution_now) - rhs
                r = r_start
                for r in range(r_start - 1, 0, -1):
                    res -= block_A_k.block_local_product(XAX[k], XAX[k + 1],
                                                np.reshape((u[:, None, r] @ v[None, r, :]).T,
                                                           (rx[k], block_size, N[k], rx[k + 1])))
                    if np.linalg.norm(res) / norm_rhs > max(2 * trunc_tol, block_res_new):
                        break
                r += 1
                u = np.reshape(u[:, :r].T, (r, N[k], rx[k + 1]))
                v = v[:r].T.reshape(rx[k], block_size, r)
                if amen and not last:
                    # amen enhancement
                    Axz = block_A_k.lcompressed_block_local_product(ZAX[k], XAX[k + 1], solution_now, shape=(rz[k], block_size, N[k], rx[k + 1]))
                    rhsxz = block_b_k.block_local_product(Zb[k], Xb[k + 1], 1, (rz[k], block_size, N[k], rx[k + 1]))
                    resxz = rhsxz.__isub__(Axz)
                    kr = min(kick_rank, rz[k] * block_size, N[k] * rx[k + 1])
                    uz, _ = truncated_svd(np.reshape(resxz, (rz[k] * block_size, N[k] * rx[k + 1])).T, kr)
                    uz = uz.T.reshape(kr, N[k], rx[k + 1])
                    u = np.concatenate((np.reshape(u, (r, N[k], rx[k + 1])), uz), axis=0)
                    u, R = scp.linalg.qr(u.reshape(-1, N[k]*rx[k+1]).T, mode="economic", check_finite=False, overwrite_a=True)
                    u = u.T.reshape(-1, N[k], rx[k+1])
                    v = einsum("Rdk, kr -> Rdr", v, R.T[:v.shape[-1]], optimize=[(0, 1)])
                    r = u.shape[0]

            else:
                r = min(prune_singular_vals(s, eps), r_max)
                u = np.reshape(u[:, :r].T, (r, N[k], rx[k + 1]))
                v = v[:r].T.reshape(rx[k], block_size, r)

            x_cores[k] = u
            x_cores[k - 1] = einsum('rdc,cbR->rbdR', x_cores[k - 1], v, optimize=[(0, 1)]) / scales
            rx[k] = r

            XAX[k] = {(i, j): compute_phi_bck_A(XAX[k + 1][(i, j)], x_cores[k], block_A_k[(i, j)], x_cores[k]) for (i, j) in block_A_k}

            Xb[k] = {i: compute_phi_bck_rhs(Xb[k + 1][i], block_b_k[i], x_cores[k]) for i in block_b_k}

            if amen and not last:
                kr = min(kick_rank, *resz.shape)
                uz, vz = truncated_svd(resz, kr)
                uz = uz.T.reshape(kr, N[k], rz[k + 1])
                vz = np.reshape(vz.T, (rz[k], block_size, kr))
                z_cores[k] = uz
                z_cores[k - 1] = einsum('rdc,cbR->rbdR', z_cores[k - 1], vz, optimize=[(0, 1)])
                rz[k] = uz.shape[0]

                ZAX[k] = {(i, j): compute_phi_bck_A(ZAX[k + 1][(i, j)], z_cores[k], block_A_k[(i, j)], x_cores[k]) for (i, j) in block_A_k}
                ZAX[k].update({(l, t): compute_phi_bck_A(ZAX[k + 1][(l, t)], z_cores[k], np.transpose(block_A_k[(i, j)], (0, 2, 1, 3)), x_cores[k]) for (i, j), (l, t) in block_A_k._transposes.items()})
                Zb[k] = {i: compute_phi_bck_rhs(Zb[k + 1][i], block_b_k[i], z_cores[k]) for i in block_b_k}

        else:
            x_cores[k] = np.reshape(solution_now.T, (rx[k], block_size, N[k], rx[k + 1]))/scales
            if amen and not last:
                z_cores[k] = np.reshape(resz.T, (rz[k], block_size, N[k], rz[k + 1]))

    return x_cores, z_cores, XAX, Xb, rx, local_res, local_dx, lgmres_discount, direct_solve_failure 


def _fwd_sweep(
        local_solver,
        x_cores,
        z_cores,
        XAX,
        ZAX,
        block_A,
        Xb,
        Zb,
        block_b,
        rx,
        rz,
        N,
        block_size,
        trunc_tol,
        d,
        swp,
        eps,
        r_max,
        kick_rank,
        last,
        amen,
        lgmres_discount,
        direct_solve_failure 
):
    local_res = np.inf if swp == 0 else 0
    local_dx = np.inf if swp == 0 else 0
    for k in range(d):
        block_A_k = block_A[k]
        block_b_k = block_b[k]
        if swp > 0 and not last:
            previous_solution = x_cores[k]
            solution_now, block_res_old, block_res_new, rhs, norm_rhs, lgmres_discount, direct_solve_failure = local_solver(
                XAX[k], block_A_k, XAX[k + 1], Xb[k],
                block_b_k, Xb[k + 1],
                previous_solution,
                r_max**(2/3), lgmres_discount, not direct_solve_failure 
            )

            local_res = max(local_res, block_res_old)
            dx = np.linalg.norm(solution_now - previous_solution) / np.linalg.norm(solution_now)
            local_dx = max(dx, local_dx)

            if amen:
                Az = block_A_k.compressed_block_local_product(ZAX[k], ZAX[k + 1], solution_now, shape=(rz[k], block_size, N[k], rz[k + 1]))
                rhsz = block_b_k.block_local_product(Zb[k], Zb[k + 1], 1, (rz[k], block_size, N[k], rz[k + 1]))
                resz = np.transpose(rhsz.__isub__(Az), (0, 2, 1, 3)).reshape(rz[k] * N[k], block_size * rz[k + 1])

            scales = np.array([np.linalg.norm(solution_now[:, k]) for k in range(solution_now.shape[1])]).reshape(1, -1, 1, 1)
            solution_now = np.transpose(scales*solution_now, (0, 2, 1, 3))
            solution_now = np.reshape(solution_now, (rx[k] * N[k], block_size * rx[k + 1]))
        else:
            solution_now = x_cores[k]
            scales = np.array([np.linalg.norm(solution_now[:, k]) for k in range(solution_now.shape[1])]).reshape(1, -1, 1, 1)
            solution_now = (scales*solution_now).transpose(0, 2, 1, 3)
            solution_now = np.reshape(solution_now, (rx[k] * N[k],  block_size * rx[k + 1]))
            if amen and not last:
                resz = np.reshape(z_cores[k].transpose(0, 2, 1, 3), (rz[k] * N[k], block_size * rz[k + 1]))


        if k < d - 1:
            if min(rx[k] * N[k],  block_size * rx[k + 1]) > 2*r_max:
                u, s, v = scp.sparse.linalg.svds(solution_now, k=r_max, tol=eps, which="LM")
                idx = np.argsort(s)[::-1]  # descending order
                s = s[idx]
                u = u[:, idx]
                v = v[idx, :]
            else:
                u, s, v = scp.linalg.svd(solution_now, full_matrices=False, check_finite=False, overwrite_a=True)
            v = s.reshape(-1, 1) * v
            u = u.reshape(rx[k], N[k], -1)
            v = v.reshape(-1, block_size, rx[k + 1])

            if swp > 0 and not last:
                r_start = min(prune_singular_vals(s, eps), r_max)
                solution_now = einsum("rbR, Rdk -> rbdk", u[:, :, :r_start], v[:r_start], optimize=[(0, 1)])
                res = block_A_k.block_local_product(XAX[k], XAX[k + 1], np.transpose(solution_now, (0, 2, 1, 3))) - rhs
                r = r_start
                for r in range(r_start - 1, 0, -1):
                    res -= block_A_k.block_local_product(XAX[k], XAX[k + 1], einsum("rbR, Rdk -> rdbk", u[:, :, None, r], v[None, r], optimize=[(0, 1)]))
                    if np.linalg.norm(res) / norm_rhs > max(2 * trunc_tol, block_res_new):
                        break
                r += 1
                if amen:
                    # amen enhancement
                    Axz = block_A_k.rcompressed_block_local_product(XAX[k], ZAX[k + 1], einsum("rbR, Rdk -> rdbk", u[:, :, :r], v[:r], optimize=[(0, 1)]), shape=(rx[k], block_size, N[k], rz[k + 1]))
                    rhsxz = block_b_k.block_local_product(Xb[k], Zb[k + 1], 1, (rx[k], block_size, N[k], rz[k + 1]))
                    resxz = np.transpose(rhsxz.__isub__(Axz), (0, 2, 1, 3))
                    kr = min(kick_rank, rx[k] * N[k], block_size * rz[k + 1])
                    uz, _ = truncated_svd(np.reshape(resxz, (rx[k] * N[k], block_size * rz[k + 1])), kr)
                    uz = np.reshape(uz, (rx[k], N[k], kr))
                    u = np.concatenate((u[:, :, :r], uz), axis=-1)
                    u, R = scp.linalg.qr(u.reshape(rx[k]*N[k], -1), mode="economic", check_finite=False, overwrite_a=True)
                    u = u.reshape(rx[k], N[k], -1)
                    v = einsum("rR, Rdk -> rdk", R[:, :r], v[:r], optimize=[(0, 1)])
                    r = v.shape[0]
                else:
                    u = u[:, :, :r]
                    v = v[:r]
            else:
                r = min(prune_singular_vals(s, eps), r_max)
                u = u[:, :, :r]
                v = v[:r]

            v = einsum("rbR, Rdk -> rbdk", v, x_cores[k + 1], optimize=[(0, 1)])
            x_cores[k] = u
            x_cores[k + 1] = v.reshape(r, block_size, N[k + 1], rx[k + 2])/scales
            rx[k + 1] = r

            XAX[k + 1] = {(i, j): compute_phi_fwd_A(XAX[k][(i, j)], x_cores[k], block_A_k[(i, j)], x_cores[k]) for
                          (i, j) in block_A_k}
            Xb[k + 1] = {i: compute_phi_fwd_rhs(Xb[k][i], block_b_k[i], x_cores[k]) for i in block_b_k}

            if amen and not last:
                kr = min(kick_rank, *resz.shape)
                uz, vz = truncated_svd(resz, kr)
                uz = np.reshape(uz, (rz[k], N[k], kr))
                vz = np.reshape(vz, (kr, block_size, rz[k + 1]))
                z_cores[k] = uz
                z_cores[k + 1] = einsum("rbR, Rdk -> rbdk", vz, z_cores[k + 1], optimize=[(0, 1)])
                rz[k + 1] = uz.shape[-1]

                ZAX[k + 1] = {(i, j): compute_phi_fwd_A(ZAX[k][(i, j)], z_cores[k], block_A_k[(i, j)], x_cores[k]) for (i, j) in block_A_k}
                ZAX[k + 1].update({(l, t): compute_phi_fwd_A(ZAX[k][(l, t)], z_cores[k], np.transpose(block_A_k[(i, j)], (0, 2, 1, 3)), x_cores[k]) for (i, j), (l, t) in block_A_k._transposes.items()})
                Zb[k + 1] = {i: compute_phi_fwd_rhs(Zb[k][i], block_b_k[i], z_cores[k]) for i in block_b_k}

        else:
            x_cores[k] = np.reshape(solution_now, (rx[k], N[k], block_size, rx[k + 1])).transpose(0, 2, 1, 3)/scales
            if amen and not last:
                z_cores[k] = np.reshape(resz, (rz[k], N[k], block_size, rz[k + 1])).transpose(0, 2, 1, 3)


    return x_cores, z_cores, XAX, Xb, rx, local_res, local_dx, lgmres_discount, direct_solve_failure 


def tt_block_amen(block_A, block_b, term_tol, r_max=100, eps=1e-12, nswp=22, x0=None, local_solver=None, kick_rank=2, amen=False, verbose=False):

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
    final_local_res = np.inf 
    lgmres_discount = 0.01
    direct_solve_failure = False
    trunc_tol = term_tol / np.sqrt(d)

    for swp in range(nswp):
        if direction > 0:
            x_cores, z_cores, XAX, Xb, rx, local_res, local_dx, lgmres_discount, direct_solve_failure = _bck_sweep(
                local_solver,
                x_cores,
                z_cores,
                XAX,
                ZAX,
                block_A,
                Xb,
                Zb,
                block_b,
                rx,
                rz,
                N,
                block_size,
                trunc_tol,
                d,
                swp,
                eps,
                r_max,
                kick_rank,
                last,
                amen,
                lgmres_discount,
                direct_solve_failure 
            )
        else:
            x_cores, z_cores, XAX, Xb, rx, local_res, local_dx, lgmres_discount, direct_solve_failure = _fwd_sweep(
                local_solver,
                x_cores,
                z_cores,
                XAX,
                ZAX,
                block_A,
                Xb,
                Zb,
                block_b,
                rx,
                rz,
                N,
                block_size,
                trunc_tol,
                d,
                swp,
                eps,
                r_max,
                kick_rank,
                last,
                amen,
                lgmres_discount,
                direct_solve_failure 
            )

        if last:
            break
        if local_res < term_tol or local_dx < eps or swp == nswp - 2:
            last = True
            final_local_res = local_res

        if verbose:
            print("\t===Finishing up===" if last else f"\t=====Sweep {swp+1}=====")
            print(f'\tDirection {direction}')
            print(f'\tResidual {local_res:.3e}')
            print(f"\tTT-sol rank: {tt_ranks(x_cores)}")
            print(f'\tLGMRES-discount: {lgmres_discount:2f}', flush=True)

        direction *= -1


    if verbose:
        print("\n\t---Results---")
        print('\tSolution rank is', rx[1:-1])
        print(f'\tResidual {final_local_res:.3e}', )
        print('\tNumber of sweeps', swp)
        print(f'\tTime: {time.time() - t0:3f}s')
        print(f'\tTime per sweep: {(time.time() - t0) / (swp+1):3f}s', flush=True)

    return x_cores, final_local_res

def _default_local_solver(XAX_k, block_A_k, XAX_k1, Xb_k, block_b_k, Xb_k1, previous_solution, size_limit, lgmres_discount, dense_solve=True, rtol=1e-10):
    x_shape = previous_solution.shape
    block_size = x_shape[1]
    m = x_shape[0]*x_shape[2]*x_shape[3]
    rhs = np.empty_like(previous_solution)
    x_shape = (x_shape[1], x_shape[0], x_shape[2], x_shape[3])
    for i in block_b_k:
        rhs[:, i] = cached_einsum('br,bmB,BR->rmR', Xb_k[i], block_b_k[i], Xb_k1[i])
    norm_rhs = max(np.linalg.norm(rhs), 1e-10)
    block_res_old = np.linalg.norm(
        block_A_k.block_local_product(XAX_k, XAX_k1, previous_solution) - rhs) / norm_rhs
    
    dense_solve = (np.sqrt(x_shape[0]*x_shape[3]) <= size_limit) and dense_solve
    direct_solve_failure = False

    if dense_solve:
        try:
            B_blocks = [[None for _ in range(block_size)] for _ in range(block_size)]

            for (i, j) in block_A_k:
                local_B = cached_einsum('lsr,smnS,LSR->lmLrnR',
                                        XAX_k[i, j], block_A_k[i, j], XAX_k1[i, j]).reshape(m, m)
                local_B_sparse = scp.sparse.csc_matrix(local_B)

                B_blocks[i][j] = local_B_sparse

                if (i, j) in block_A_k._transposes:
                    k, t = block_A_k._transposes[i, j]
                    B_blocks[k][t] = local_B_sparse.T

                if (i, j) in block_A_k._aliases:
                    k, t = block_A_k._aliases[i, j]
                    B_blocks[k][t] = local_B_sparse

            B_sparse = scp.sparse.bmat(B_blocks, format='csc')
            rhs_reshaped = np.transpose(rhs, (1, 0, 2, 3)).reshape(-1)
            x = scp.sparse.linalg.spsolve(B_sparse, rhs_reshaped)
            solution_now = x.reshape(*x_shape).transpose(1, 0, 2, 3)
        except Exception as e:
            print(f"\tAttention: {e}")
            direct_solve_failure = True

    if not dense_solve or direct_solve_failure:
        def mat_vec(x_vec):
            return np.transpose(block_A_k.block_local_product(
                XAX_k, XAX_k1,
                np.transpose(x_vec.reshape(*x_shape), (1, 0, 2, 3))
            ), (1, 0, 2, 3)).reshape(-1, 1)

        linear_op = scp.sparse.linalg.LinearOperator((block_size * m, block_size * m), matvec=mat_vec)
        if block_res_old  >= 1:
            solution_now, info = scp.sparse.linalg.lgmres(linear_op, np.transpose(rhs, (1, 0, 2, 3)).reshape(-1, 1), rtol=rtol, outer_k=5, inner_m=int(np.ceil(lgmres_discount*(block_size * m))), maxiter=100)
            solution_now = np.transpose(solution_now.reshape(*x_shape), (1, 0, 2, 3))
        else:
            solution_now, info = scp.sparse.linalg.lgmres(linear_op, np.transpose(
                rhs - block_A_k.block_local_product(XAX_k, XAX_k1, previous_solution), (1, 0, 2, 3)).reshape(-1, 1), rtol=rtol, outer_k=5, inner_m=int(np.ceil(lgmres_discount*(block_size * m))), maxiter=100)
            solution_now = np.transpose(solution_now.reshape(*x_shape), (1, 0, 2, 3))
            solution_now += previous_solution

    block_res_new = np.linalg.norm(block_A_k.block_local_product(XAX_k, XAX_k1, solution_now) - rhs) / norm_rhs

    if not dense_solve:
        score = 1.1 if block_res_new/rtol > 1 else 0.99
        lgmres_discount = max(min(0.5, lgmres_discount*score), 1e-3)


    if block_res_old < block_res_new:
        solution_now = previous_solution

    return solution_now, block_res_old, min(block_res_old, block_res_new), rhs, norm_rhs, lgmres_discount, direct_solve_failure


def tt_restarted_block_amen(
    block_A,
    block_b,
    rank_restriction,
    op_tol,
    termination_tol=1e-3,
    eps=1e-12,
    num_restarts=3,
    inner_m=10,
    x0=None,
    local_solver=None,
    refinement=False,
    verbose=False
):
    if x0 is not None:
        dim = len(x0)
        x0 = tt_rank_retraction(x0, [dim]*(dim-1))

    def solve_als(rhs, rank, x0, refinement):
        return tt_block_amen(
            block_A, rhs, termination_tol, r_max=rank, eps=eps, nswp=inner_m, x0=x0, local_solver=local_solver, kick_rank=2 + 2*refinement, amen=True, verbose=verbose
        )

    def update_rhs(rhs, x_cores):
        Ax = block_A.block_product(x_cores, 0.1 * op_tol)
        return rhs - Ax

    if verbose:
        print("\n\tStarting Restarted TT-ALS.")
        for (i, j) in block_A:
            print(f"A[{i, j}]: {tt_ranks(block_A[i, j])}")
        for i in block_b:
            print(f"b[{i}]: {tt_ranks(block_b.get_row(i))}")

    rhs = block_b
    orig_rhs_norm = rhs.norm

    if orig_rhs_norm < 0.5*op_tol:
        raise RuntimeError(f"\n\tAbsolute tolerance already reached: {orig_rhs_norm:4f} < {op_tol:4f}")

    # === First ALS solve ===
    x_cores, res = solve_als(rhs, rank_restriction, x0, False)

    if res < termination_tol:
        if verbose:
            print(f"\n\tTerminated on local criterion, Relative Error < {termination_tol:4f}")
        return x_cores, res

    # === Update RHS and check for early stopping ===
    rhs = update_rhs(rhs, x_cores)
    rhs_norm = rhs.norm

    if rhs_norm < termination_tol * orig_rhs_norm:
        if verbose:
            print(f"\n\tTerminated on global criterion, Relative Error = {rhs_norm / orig_rhs_norm:.3e}")
        return x_cores, res
    if rhs_norm > orig_rhs_norm:
        raise RuntimeError(f"Terminated on instability: ||rhs|| = {rhs_norm:4f} > previous = {orig_rhs_norm:4f}")

    if verbose:
        print(f"\n\tRelative Error = {rhs_norm / orig_rhs_norm:4f}")

    # === Restart loop ===
    for i in range(1, num_restarts):
        if rhs_norm / orig_rhs_norm > 0.5:
            inner_m += 1

        if verbose:
            print(f"\n\t--- Restart {i}")

        new_x_cores, res = solve_als(rhs, rank_restriction, None, True)

        rhs = update_rhs(rhs, new_x_cores)
        prev_rhs_norm, rhs_norm = rhs_norm, rhs.norm

        if rhs_norm >= prev_rhs_norm:
            if prev_rhs_norm >= orig_rhs_norm:
                raise RuntimeError(f"Terminated on instability: ||rhs|| = {prev_rhs_norm:4f} > previous = {orig_rhs_norm:4f}")
            if verbose:
                print(f"\n\tTerminated on instability: ||rhs|| = {rhs_norm:4f} > previous = {prev_rhs_norm:4f}")
            return x_cores, prev_rhs_norm

        if rhs_norm < termination_tol * orig_rhs_norm:
            if verbose:
                print(f"\n\tTerminated on global criterion, Relative Error = {rhs_norm / orig_rhs_norm:4f}")
            x_cores = tt_rank_reduce_py(tt_add(x_cores, new_x_cores), eps=eps)
            break

        if verbose:
            print(f"\n\tRelative Error = {rhs_norm / orig_rhs_norm:4f}")
        x_cores = tt_rank_reduce_py(tt_add(x_cores, new_x_cores), eps=eps)

    else:
        if verbose:
            print(f"\n\tNumber of restarts exhausted, Relative Error = {rhs_norm / orig_rhs_norm:4f}")

    return x_cores, res


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


class CgIterInv(scp.sparse.linalg.LinearOperator):

    def __init__(self, M, tol=1e-12):
        self.M = M
        self.shape = M.shape
        self.tol = tol
        self.ifunc = lambda b: scp.sparse.linalg.cg(M, b, maxiter=500, rtol=tol)

    def _matvec(self, x):
        b, info = self.ifunc(x)
        if info < 0:
            raise ValueError("Error in inverting M: function "
                             "%s did not converge (info = %i)."
                             % (self.ifunc.__name__, info))
        return b


class SpCholInv(scp.sparse.linalg.LinearOperator):
    def __init__(self, M):
        M = M.tocsc()
        self.shape = M.shape
        self.dtype = M.dtype

        # CHOLMOD needs symmetric matrix
        self.factor = sparse_cholesky(M)

    def _matvec(self, x):
        return self.factor.solve_A(x)


def _step_size_local_solve(previous_solution, XDX_k, Delta_k, XDX_k1, XAX_k, A_k, XAX_k1, dense_solve, step_size, eps, lanczos_discount):
    m = np.prod(previous_solution.shape)
    if dense_solve:
        previous_solution = previous_solution.reshape(-1, 1)
        D = scp.sparse.csr_matrix(cached_einsum(
            "lsr,smnS,LSR->lmLrnR",
            XDX_k, Delta_k, XDX_k1
        ).reshape(m, m))
        A = scp.sparse.csr_matrix(cached_einsum("lsr,smnS,LSR->lmLrnR", XAX_k, A_k, XAX_k1).reshape(m, m))
        try:
            eig_val, solution_now = scp.sparse.linalg.eigsh((1/step_size)*A + D, tol=eps, k=1, ncv=max(int(np.floor(lanczos_discount*m)), min(m, 5)), maxiter=10*m, which="SA", v0=previous_solution)
        except Exception as e:
            print(f"\tAttention: {e}")
            eig_val = previous_solution.T @ ((1/step_size)*A + D)  @ previous_solution
            solution_now = previous_solution
            lanczos_discount = min(0.999, lanczos_discount*1.1)
        if eig_val < 0:
            try:
                Minv = SpCholInv(A)
                eig_val, solution_now = scp.sparse.linalg.eigsh(-D, M=A, Minv=Minv, tol=eps, k=1, ncv=max(int(np.floor(lanczos_discount*m)), min(m, 5)), which="LA", maxiter=10*m, v0=previous_solution)
                step_size = max(0, min(step_size, 1/ eig_val[0]))
            except Exception as e:
                print(f"\tAttention: {e}")
                solution_now = previous_solution
                lanczos_discount = min(0.999, lanczos_discount*1.1)

        eig_val = previous_solution.T @ ((1/step_size)*A + D) @ previous_solution
        old_res = np.linalg.norm(((1/step_size)*A + D) @ previous_solution - eig_val*previous_solution)
    else:
        x_shape = previous_solution.shape
        previous_solution = previous_solution.reshape(-1, 1)
        # 'lsr,smnk,LSR,rnR-> lmkLS' 'ks'
        _mat_vec_A = contract_expression('lsr,smnS,LSR,rnR->lmL', XAX_k.shape, A_k.shape, XAX_k1.shape, x_shape, optimize="greedy")
        mat_vec_A = lambda x_vec: _mat_vec_A(XAX_k, A_k, XAX_k1, x_vec.reshape(*x_shape)).reshape(-1, 1).__iadd__(1e-12*x_vec.reshape(-1, 1)) # regularisation term for convergence
        A_op = scp.sparse.linalg.LinearOperator((m, m), matvec=mat_vec_A)
        _mat_vec_D = contract_expression('lsr,smnS,LSR,rnR->lmL', XDX_k.shape, Delta_k.shape, XDX_k1.shape, x_shape, optimize="greedy")
        mat_vec_D = lambda x_vec: _mat_vec_D(XDX_k, Delta_k, XDX_k1, x_vec.reshape(*x_shape)).reshape(-1, 1).__imul__(-1)
        D_op = scp.sparse.linalg.LinearOperator((m, m), matvec=mat_vec_D)
        AD_op = scp.sparse.linalg.LinearOperator((m, m), matvec=lambda x_vec: (mat_vec_A(x_vec) / step_size).__isub__(mat_vec_D(x_vec)))

        try:
            eig_val, solution_now = scp.sparse.linalg.eigsh(AD_op, tol=eps, k=1, ncv=max(int(np.floor(lanczos_discount*m)), min(m, 5)), which="SA", maxiter=10*m, v0=previous_solution)
        except Exception as e:
            eig_val = previous_solution.T @ AD_op(previous_solution)
            solution_now = previous_solution
            lanczos_discount = min(0.999, lanczos_discount*1.1)
        if eig_val < 0:
            try:
                Minv = CgIterInv(A_op, tol=eps)
                eig_val, solution_now = scp.sparse.linalg.eigsh(D_op, M=A_op, Minv=Minv, tol=eps, k=1, ncv=max(int(np.floor(lanczos_discount*m)), min(m, 5)), which="LA", maxiter=10*m, v0=previous_solution)
                step_size = max(0, min(step_size, 1 / eig_val[0]))
            except Exception as e:
                print(f"\tAttention: {e}")
                solution_now = previous_solution
                lanczos_discount = min(0.999, lanczos_discount*1.1)

        eig_val = previous_solution.T @ AD_op(previous_solution)
        old_res = np.linalg.norm(AD_op(previous_solution).__isub__(eig_val * previous_solution))
    lanczos_discount = max(0.1, lanczos_discount*0.999)
    return solution_now.reshape(-1, 1), step_size, old_res, lanczos_discount


def _local_psd_check(previous_solution, XAX_k, A_k, XAX_k1, dense_solve, eps):
    m = np.prod(previous_solution.shape)
    if dense_solve:
        try:
            eig_val, _ = scp.sparse.linalg.eigsh(cached_einsum("lsr,smnS,LSR->lmLrnR", XAX_k, A_k, XAX_k1).reshape(m, m), tol=eps, k=1, which="SA")
        except:
            eig_val = -1
    else:
        x_shape = previous_solution.shape
        _mat_vec_A = contract_expression('lsr,smnS,LSR,rnR->lmL', XAX_k.shape, A_k.shape, XAX_k1.shape, x_shape, optimize="greedy")
        mat_vec_A = lambda x_vec: _mat_vec_A(XAX_k, A_k, XAX_k1, x_vec.reshape(*x_shape)).reshape(-1, 1)
        A_op = scp.sparse.linalg.LinearOperator((m, m), matvec=mat_vec_A)
        try:
            eig_val, _ = scp.sparse.linalg.eigsh(A_op, tol=eps, k=1, which="SA")
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
        print(f"\nStarting Eigen solve with:\n \t {tol} \n \t sweeps: {nswp}")
        t0 = time.time()
    if x0 is None:
        x_cores = tt_random_gaussian([2]*(len(A)-1), (A[0].shape[2],))
        if kick_rank is None:
            kick_rank = np.maximum(np.ceil(symmetric_powers_of_two(len(A)-1) / (nswp-2)), 2).astype(int)
    else:
        x_cores = x0
        if kick_rank is None:
            kick_rank = np.maximum(np.ceil((symmetric_powers_of_two(len(A)-1) - np.array(tt_ranks(x_cores))) / (nswp-2)), 2).astype(int)

    d = len(x_cores)
    rx = np.array([1] + tt_ranks(x_cores) + [1])
    N = np.array([c.shape[1] for c in x_cores])

    XAX = [np.ones((1, 1, 1))] + [None] * (d - 1) + [np.ones((1, 1, 1))]  # size is rk x Rk x rk
    XDX = [np.ones((1, 1, 1))] + [None] * (d - 1) + [np.ones((1, 1, 1))]

    step_size = 1
    lanczos_discount = 0.5
    last = False
    size_limit = (2**(d/2))**(3/4)
    local_res = np.inf*np.ones((2, d-1))
    for swp in range(nswp):
        for k in range(d - 1, -1, -1):
            if swp > 0:
                previous_solution = x_cores[k]
                solution_now, step_size, res, lanczos_discount = _step_size_local_solve(previous_solution, XDX[k], Delta[k], XDX[k+1], XAX[k], A[k], XAX[k+1], np.sqrt(rx[k] * rx[k + 1]) < size_limit, step_size, tol, lanczos_discount)

                if 2*local_res[0, k-1] < res:
                    if not _local_psd_check(previous_solution, XAX[k], A[k], XAX[k+1], np.sqrt(rx[k] * rx[k + 1]) < size_limit, tol):
                        print(f"\t Matrix A is not positive definite!", flush=True)
                        break
                else:
                    local_res[0, k-1] = res
                solution_now = np.reshape(solution_now, (rx[k], N[k] * rx[k + 1])).T
            else:
                solution_now = np.reshape(x_cores[k], (rx[k], N[k] * rx[k + 1])).T

            if k > 0:
                u, s, v = scp.linalg.svd(solution_now, full_matrices=False, check_finite=False, overwrite_a=True, lapack_driver="gesvd")
                v = s.reshape(-1, 1) * v
                r = prune_singular_vals(s, 0.1*tol)
                if not last:
                    kick = kick_rank[k-1]
                    u, v, r = _add_kick_rank(u[:, :r], v[:r], kick)
                else:
                    u = u[:, :r]
                    v = v[:r]
                x_cores[k] = np.reshape(u.T, (r, N[k], rx[k + 1]))
                x_cores[k - 1] = einsum('rdc,cR->rdR', x_cores[k - 1], v.T, optimize=[(0, 1)])
                rx[k] = r

                XAX[k] = compute_phi_bck_A(XAX[k + 1], x_cores[k], A[k], x_cores[k])
                XDX[k] = compute_phi_bck_A(XDX[k + 1], x_cores[k], Delta[k], x_cores[k])
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
            print(f'\tResidual {np.max(local_res[0])}')
            print(f'\tLanczos Discount {lanczos_discount:2f}')
            print(f"\tTT-sol rank: {tt_ranks(x_cores)}", flush=True)
        for k in range(d):
            previous_solution = x_cores[k]
            solution_now, step_size, res, lanczos_discount = _step_size_local_solve(previous_solution, XDX[k], Delta[k], XDX[k+1], XAX[k], A[k], XAX[k+1], np.sqrt(rx[k] * rx[k + 1]) < size_limit, step_size, tol, lanczos_discount)
            if 2*local_res[1, k-1] < res:
                if not _local_psd_check(previous_solution, XAX[k], A[k], XAX[k + 1], np.sqrt(rx[k] * rx[k + 1]) < size_limit, tol):
                    print(f"\t Matrix A is not positive definite!", flush=True)
                    break
            else:
                local_res[1, k-1] = res
            solution_now = np.reshape(solution_now, (rx[k] * N[k], rx[k + 1]))
            if k < d - 1:
                u, s, v = scp.linalg.svd(solution_now, full_matrices=False, check_finite=False, overwrite_a=True, lapack_driver="gesvd")
                v = s.reshape(-1, 1) * v
                r = prune_singular_vals(s, 0.1*tol)
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
            print(f'\tResidual {np.max(local_res[1])}')
            print(f'\tLanczos Discount {lanczos_discount:2f}')
            print(f"\tTT-sol rank: {tt_ranks(x_cores)}", flush=True)

    max_res  = min(np.max(local_res[0]), np.max(local_res[1]))
    if verbose:
        print("\t -----")
        print(f"\t Solution rank is {rx[1:-1]}")
        print('\t Step size: %f' % step_size)
        print(f"\t Residual {max_res}")
        print('\t Number of sweeps', swp + 1)
        print('\t Time: ', time.time() - t0)
        print('\t Time per sweep: ', (time.time() - t0) / (swp + 1), flush=True)

    if max_res > 10*tol:
        print('\t Target Residual not reached!', flush=True)
        step_size = 0
    return step_size, x_cores


def tt_min_eig(A, x0=None, kick_rank=None, nswp=10, tol=1e-12, verbose=False):
    if verbose:
        print(f"\nStarting Eigen solve with:\n \t {tol} \n \t sweeps: {nswp}")
        t0 = time.time()
    if x0 is None:
        x_cores = tt_random_gaussian([2]*(len(A)-1), (A[0].shape[2],))
    else:
        x_cores = tt_rank_retraction(x0, [2]*(len(A)-1))
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
                u, s, v = scp.linalg.svd(solution_now, full_matrices=False, check_finite=False, overwrite_a=True, lapack_driver="gesvd")
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

                XAX[k] = compute_phi_bck_A(XAX[k + 1], x_cores[k], A[k], x_cores[k])
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
                u, s, v = scp.linalg.svd(solution_now, full_matrices=False, check_finite=False, overwrite_a=True, lapack_driver="gesvd")
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
        print('\t Time per sweep: ', (time.time() - t0) / (swp + 1),flush=True)

    min_eig_value = tt_inner_prod(x_cores, tt_fast_matrix_vec_mul(A, x_cores, tol))
    return x_cores, min_eig_value


def _eigen_local_solve(previous_solution, XAX_k, A_k, XAX_k1, m, size_limit, eps):
    if m <= size_limit:
        previous_solution = previous_solution.reshape(-1, 1)
        A = cached_einsum("lsr,smnS,LSR->lmLrnR", XAX_k, A_k, XAX_k1).reshape(m, m)
        try:
            eig_val, solution_now = scp.sparse.linalg.eigsh(A, tol=eps, k=1, which="SA", v0=previous_solution)
        except:
            solution_now = previous_solution
            eig_val = previous_solution.T @ A @ previous_solution
        old_res = np.linalg.norm(eig_val * previous_solution - A @ previous_solution)
        return solution_now, old_res

    x_shape = previous_solution.shape
    previous_solution = previous_solution.reshape(-1, 1)
    _mat_vec_A = contract_expression('lsr,smnS,LSR,rnR->lmL', XAX_k.shape, A_k.shape, XAX_k1.shape, x_shape, optimize="greedy")
    mat_vec_A = lambda x_vec: _mat_vec_A(XAX_k, A_k, XAX_k1, x_vec.reshape(*x_shape)).reshape(-1, 1)
    A_op = scp.sparse.linalg.LinearOperator((m, m), matvec=mat_vec_A)
    try:
        eig_val, solution_now = scp.sparse.linalg.eigsh(A_op, tol=eps, k=1, which="SA", v0=previous_solution)
    except:
        solution_now = previous_solution
        eig_val = previous_solution.T @ A_op(previous_solution)

    old_res = np.linalg.norm(eig_val * previous_solution - A_op(previous_solution))
    return solution_now.reshape(-1, 1), old_res


def tt_approx_mat_mat_mul(A, D, x0=None, kick_rank=None, nswp=50, tol=1e-6, verbose=False):
    if verbose:
        print(f"\nStarting MM solve with:\n \t {tol} \n \t sweeps: {nswp}", flush=True)
        t0 = time.time()
    if x0 is None:
        max_ranks = np.maximum((np.array(tt_ranks(A)) + np.array(tt_ranks(D))) / 2, 2).astype(int)
        x_cores = tt_random_gaussian(list(max_ranks), A[0].shape[1:-1])
    else:
        x_cores = x0
        max_ranks = np.array(tt_ranks(x0))

    if kick_rank is None:
        kick_rank = np.maximum(((symmetric_powers_of_two(len(A)-1) - max_ranks) / (nswp / 2)), 2).astype(int)

    d = len(x_cores)
    rx = np.array([1] + tt_ranks(x_cores) + [1])
    N = np.array([c.shape[1] for c in x_cores])
    M = np.array([c.shape[2] for c in x_cores])

    XADX = [np.ones((1, 1, 1))] + [None] * (d - 1) + [np.ones((1, 1, 1))]  # size is rk x Rk x rk

    normAD = np.ones(d - 1)  # norm of each row in the block matrix
    nrmsc = 1.0
    normx = np.ones((d - 1))
    tol = tol / np.sqrt(d)

    max_res = 0
    last = False
    for swp in range(nswp):
        max_res = np.inf if swp == 0 else 0
        for k in range(d - 1, -1, -1):
            if swp > 0:
                previous_solution = x_cores[k]
                solution_now = cached_einsum('rab,amkA,bknB,RAB->rmnR',XADX[k], A[k], D[k], XADX[k+1])
                solution_now *= nrmsc
                local_res = np.linalg.norm(solution_now - previous_solution) / max(np.linalg.norm(solution_now), 1e-8)
                max_res = max(max_res, local_res)
                solution_now = np.reshape(solution_now, (rx[k], N[k] * M[k] * rx[k + 1])).T
            else:
                solution_now = np.reshape(x_cores[k], (rx[k], N[k] * M[k] * rx[k + 1])).T

            if k > 0:
                u, s, v = scp.linalg.svd(solution_now, full_matrices=False, check_finite=False, overwrite_a=True, lapack_driver="gesvd")
                v = s.reshape(-1, 1) * v
                r = prune_singular_vals(s, tol)
                if not last:
                    kick = kick_rank[k-1]
                    u, v, r = add_kick_rank(u[:, :r], v[:r], kick)
                else:
                    u = u[:, :r]
                    v = v[:r]
                nrmsc *= normx[k - 1] / normAD[k - 1]
                x_cores[k] = np.reshape(u.T, (r, N[k], M[k], rx[k + 1]))
                x_cores[k - 1] = np.tensordot(x_cores[k - 1], v.T, axes=([3], [0])) # 'rdkc,cR->rdkR'
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
        max_res = 0
        for k in range(d):
            previous_solution = x_cores[k]
            solution_now = cached_einsum('rab,amkA,bknB,RAB->rmnR', XADX[k], A[k], D[k], XADX[k + 1])
            solution_now *= nrmsc
            local_res = np.linalg.norm(solution_now - previous_solution) / max(np.linalg.norm(solution_now), 1e-8)
            max_res = max(max_res, local_res)
            solution_now = np.reshape(solution_now, (rx[k] * N[k] * M[k], rx[k + 1]))
            if k < d - 1:
                nrmsc *= normx[k] / normAD[k]
                u, s, v = scp.linalg.svd(solution_now, full_matrices=False, check_finite=False, overwrite_a=True, lapack_driver="gesvd")
                v = s.reshape(-1, 1) * v
                r = prune_singular_vals(s, tol)
                if not last:
                    kick = kick_rank[k]
                    u, v, r = add_kick_rank(u[:, :r], v[:r, :], kick)
                else:
                    u = u[:, :r]
                    v = v[:r, :]
                x_cores[k] = u.reshape(rx[k], N[k], M[k], r)
                x_cores[k + 1] = np.tensordot(v, x_cores[k + 1], axes=([1], [0])).reshape(r, N[k + 1], M[k+1], rx[k + 2]) # ij,jdkl->idkl
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
            print(f'\tResidual {max_res}')
            print(f"\tTT-sol rank: {tt_ranks(x_cores)}", flush=True)

    if verbose:
        print("\t -----")
        print(f"\t Solution rank is {rx[1:-1]}")
        print(f"\t Residual {max_res}")
        print('\t Number of sweeps', swp + 1)
        print('\t Time: ', time.time() - t0)
        print('\t Time per sweep: ', (time.time() - t0) / (swp + 1), flush=True)

    normx = np.exp(np.sum(np.log(normx)) / d)

    return [normx * core for core in x_cores]


def tt_mat_mat_mul(mat1, mat2, op_tol, eps, verbose=False):
    if np.max((np.array(tt_ranks(mat1)) + np.array(tt_ranks(mat2)))/2) <= 20:
        return tt_rank_reduce(tt_fast_mat_mat_mul(mat1, mat2, eps), eps=op_tol)
    return tt_approx_mat_mat_mul(mat1, mat2, tol=op_tol, verbose=verbose)


def tt_approx_mat_vec_mul(A, d_vec, x0=None, kick_rank=None, nswp=50, tol=1e-6, verbose=False):
    if verbose:
        print(f"\nStarting MM solve with:\n \t {tol} \n \t sweeps: {nswp}")
        t0 = time.time()
    if x0 is None:
        max_ranks = np.maximum((np.array(tt_ranks(A)) + np.array(tt_ranks(d_vec))) / 2, 2).astype(int)
        x_cores = tt_random_gaussian(list(max_ranks), (A[0].shape[2], ))
    else:
        x_cores = x0
        max_ranks = np.array(tt_ranks(x0))

    if kick_rank is None:
        kick_rank = np.maximum(((symmetric_powers_of_two(len(A)-1) - max_ranks) / (nswp / 2)), 2).astype(int)

    d = len(x_cores)
    rx = np.array([1] + tt_ranks(x_cores) + [1])
    N = np.array([c.shape[1] for c in x_cores])

    XAdX = [np.ones((1, 1, 1))] + [None] * (d - 1) + [np.ones((1, 1, 1))]  # size is rk x Rk x rk

    normAd = np.ones(d - 1)  # norm of each row in the block matrix
    nrmsc = 1.0
    normx = np.ones((d - 1))
    tol = tol / np.sqrt(d)

    max_res = 0
    last = False
    for swp in range(nswp):
        max_res = np.inf if swp == 0 else 0
        for k in range(d - 1, -1, -1):
            if swp > 0:
                previous_solution = x_cores[k]
                solution_now = cached_einsum('rab,amkA,bkB,RAB->rmR', XAdX[k], A[k], d_vec[k], XAdX[k + 1])
                solution_now *= nrmsc
                local_res = np.linalg.norm(solution_now - previous_solution) / max(np.linalg.norm(solution_now), 1e-8)
                max_res = max(max_res, local_res)
                solution_now = np.reshape(solution_now, (rx[k], N[k] * rx[k + 1])).T
            else:
                solution_now = np.reshape(x_cores[k], (rx[k], N[k] * rx[k + 1])).T

            if k > 0:
                u, s, v = scp.linalg.svd(solution_now, full_matrices=False, check_finite=False, overwrite_a=True, lapack_driver="gesvd")
                v = s.reshape(-1, 1) * v
                r = prune_singular_vals(s, tol)
                if not last:
                    kick = kick_rank[k-1]
                    u, v, r = add_kick_rank(u[:, :r], v[:r], kick)
                else:
                    u = u[:, :r]
                    v = v[:r]
                nrmsc *= normx[k - 1] / normAd[k - 1]
                x_cores[k] = np.reshape(u.T, (r, N[k], rx[k + 1]))
                x_cores[k - 1] = np.tensordot(x_cores[k - 1], v.T, axes=([2], [0])) # rdc,cR->rdR
                norm_now = np.linalg.norm(x_cores[k - 1])
                normx[k - 1] *= norm_now
                x_cores[k - 1] /= norm_now
                rx[k] = r

                XAdX[k] = cached_einsum('RAB,amkA,bkB,rmR->rab', XAdX[k+1], A[k], d_vec[k], x_cores[k])
                norm = np.linalg.norm(XAdX[k])
                norm = norm if norm > 0 else 1.0
                XAdX[k] /= norm
                normAd[k-1] = norm
                nrmsc *= normAd[k - 1] / normx[k - 1]
            else:
                x_cores[k] = np.reshape(solution_now, (rx[k], N[k], rx[k + 1]))

        if last:
            break
        if max_res < tol or swp == nswp - 1:
            last = True
        max_res = 0
        for k in range(d):
            previous_solution = x_cores[k]
            solution_now = cached_einsum('rab,amkA,bkB,RAB->rmR', XAdX[k], A[k], d_vec[k], XAdX[k + 1])
            solution_now *= nrmsc
            local_res = np.linalg.norm(solution_now - previous_solution) / max(np.linalg.norm(solution_now), 1e-8)
            max_res = max(max_res, local_res)
            solution_now = np.reshape(solution_now, (rx[k] * N[k], rx[k + 1]))
            if k < d - 1:
                nrmsc *= normx[k] / normAd[k]
                u, s, v = scp.linalg.svd(solution_now, full_matrices=False, check_finite=False, overwrite_a=True, lapack_driver="gesvd")
                v = s.reshape(-1, 1) * v
                r = prune_singular_vals(s, tol)
                if not last:
                    kick = kick_rank[k]
                    u, v, r = add_kick_rank(u[:, :r], v[:r, :], kick)
                else:
                    u = u[:, :r]
                    v = v[:r, :]
                x_cores[k] = u.reshape(rx[k], N[k], r)
                x_cores[k + 1] = np.tensordot(v, x_cores[k + 1], axes=([1], [0])).reshape(r, N[k + 1], rx[k + 2]) # ij,jdl->idl
                norm_now = np.linalg.norm(x_cores[k + 1])
                normx[k] *= norm_now
                x_cores[k + 1] /= norm_now
                rx[k + 1] = r

                XAdX[k + 1] = cached_einsum('rab,amkA,bkB,rmR->RAB', XAdX[k], A[k], d_vec[k], x_cores[k])
                norm = np.linalg.norm(XAdX[k + 1])
                norm = norm if np.greater(norm, 0) else 1.0
                XAdX[k + 1] /= norm
                normAd[k] = norm
                nrmsc *= normAd[k] / normx[k]
            else:
                x_cores[k] = np.reshape(solution_now, (rx[k], N[k], rx[k + 1]))

        if last:
            break
        if max_res < tol:
            last = True
        if verbose:
            print('\tStarting Sweep: %d' % swp)
            print(f'\tResidual {max_res}')
            print(f"\tTT-sol rank: {tt_ranks(x_cores)}", flush=True)

    if verbose:
        print("\t -----")
        print(f"\t Solution rank is {rx[1:-1]}")
        print(f"\t Residual {max_res}")
        print('\t Number of sweeps', swp + 1)
        print('\t Time: ', time.time() - t0)
        print('\t Time per sweep: ', (time.time() - t0) / (swp + 1), flush=True)

    normx = np.exp(np.sum(np.log(normx)) / d)

    return [normx * core for core in x_cores]


def tt_mat_vec_mul(mat, vec, op_tol, eps, verbose=False):
    if np.max((np.array(tt_ranks(mat)) + np.array(tt_ranks(vec)))/2) <= 40:
        return tt_rank_reduce(tt_fast_matrix_vec_mul(mat, vec, eps), op_tol)
    return tt_approx_mat_vec_mul(mat, vec, tol=op_tol, verbose=verbose)

