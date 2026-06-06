import sys
import os
import copy
import time
import numpy as np
import traceback
import scipy.linalg as la

sys.path.append(os.getcwd() + '/../')

from src.tt_ops import *
from src.tt_als import TTBlockMatrix, TTBlockVector, tt_mat_vec_mul, tt_max_generalised_eigen, tt_min_eig, tt_mat_mat_mul,tt_restarted_block_amen
from cy_src.lgmres_cy import (
    core_matvec,
    core_rmatvec,
    core_sum_contract,
    dense_core_matrix,
    dense_schur_3block,
    dense_schur_4block,
    rhs_contract,
)
from dataclasses import dataclass
from enum import Enum
from petsc4py import PETSc

import warnings
warnings.simplefilter("error")



def chunk_integer(n, k):
    base_size = n // k
    remainder = n % k
    chunk_sizes = [base_size + 1 if i < remainder else base_size for i in range(k)]
    indices = np.cumsum([0] + chunk_sizes)
    return indices


class ApproxBlockLZInv:
    def __init__(self, XAX_k_21, block_A_k_21, XAX_k1_21, indices, eps=1e-11):
        self.indices = list(zip(indices[:-1], indices[1:]))
        self.base_size = block_A_k_21.shape[1] * XAX_k1_21.shape[0]

        self.inv_blocks = [
            la.cholesky(
                dense_core_matrix(
                    XAX_k_21[r_i:r_ip1, :, r_i:r_ip1], block_A_k_21, XAX_k1_21
                )
                + eps * np.eye((r_ip1-r_i)*self.base_size)
            )
            for r_i, r_ip1 in self.indices
        ]

    def solve(self, x):
        y_blocks = [
            la.cho_solve((L, True), x[r_i*self.base_size:r_ip1*self.base_size])
            for L, (r_i, r_ip1) in zip(self.inv_blocks, self.indices)
        ]
        return np.concatenate(y_blocks)


class ApproxBlockKyInv:
    def __init__(self, XAX_k_00, block_A_k_00, XAX_k1_00, indices, eps=1e-11):
        self.indices = list(zip(indices[:-1], indices[1:]))
        self.base_size = block_A_k_00.shape[1] * XAX_k1_00.shape[0]

        self.inv_blocks = [
            la.lu_factor(
                dense_core_matrix(
                    XAX_k_00[r_i:r_ip1, :, r_i:r_ip1], block_A_k_00, XAX_k1_00
                )
                + eps * np.eye((r_ip1-r_i)*self.base_size)
            )
            for r_i, r_ip1 in self.indices
        ]

    def solve(self, x):
        y_blocks = [
            la.lu_solve(lu, x[r_i*self.base_size:r_ip1*self.base_size])
            for lu, (r_i, r_ip1) in zip(self.inv_blocks, self.indices)
        ]
        return np.concatenate(y_blocks)


class BlockJacPrec:
    def __init__(
        self,
        XAX_k_00, block_A_k_00, XAX_k1_00,
        XAX_k_21, block_A_k_21, XAX_k1_21,
        nblocks,
        eps=1e-11
    ):
        x_shape = (XAX_k_00.shape[-1], block_A_k_00.shape[2], XAX_k1_00.shape[-1])
        nblocks = min(nblocks, XAX_k_00.shape[-1])
        indices = chunk_integer(x_shape[0], nblocks)
        self.m = np.prod(x_shape)
        self.KyInv = ApproxBlockKyInv(XAX_k_00, block_A_k_00, XAX_k1_00, indices, eps)
        self.LZInv = ApproxBlockLZInv(XAX_k_21, block_A_k_21, XAX_k1_21, indices, eps)

    def apply(self, _, x, y):
        x_np = x.array_r
        x1, x2 = x_np[:self.m], x_np[self.m:]
        y2 = self.LZInv.solve(x2)
        y1 = self.KyInv.solve(x1)
        y_np = np.concatenate([y1, y2])
        y.array_w[:] = y_np

class LGMRESSolver:
    def __init__(self, rtol=1e-8, max_iter=300, restart=100, outer_k=10):
        """
        Initializes the LGMRES solver.

        Args:
            matvec_object: An object with a method matvec(x) that returns A @ x,
                           where x is a NumPy array.
            shape: A tuple (N, N) representing the shape of the linear operator A.
            rtol: The relative tolerance for convergence.
            max_iter: The maximum number of iterations.
            restart: The number of iterations before GMRES restarts (inner iterations).
            outer_k: The number of outer vectors to use for LGMRES augmentation.
        """
        self.matvec_object = None
        self.shape = None
        self.x_buffer = None

        # PETSc solver setup
        self.ksp = PETSc.KSP().create(comm=PETSc.COMM_WORLD)
        self.ksp.setType('lgmres')
        opts = PETSc.Options()
        opts.setValue('-ksp_lgmres_augment', outer_k)
        #opts.setValue('-ksp_dgmres_eigen', outer_k)
        opts.setValue('-ksp_rtol', rtol)
        opts.setValue('-ksp_max_it', max_iter)
        opts.setValue('-ksp_gmres_restart', restart)
        self.ksp.setFromOptions()

    def mult(self, _, x, y):
        x_np = x.array_r
        if x_np.dtype != np.float64 or not x_np.flags.c_contiguous:
            x_np = np.ascontiguousarray(x_np, dtype=np.float64)
        y_np = y.array_w
        if y_np.dtype == np.float64 and y_np.flags.c_contiguous:
            self.matvec_object.matvec_into(x_np, y_np)
        else:
            y_np[:] = self.matvec_object.matvec(x_np)

    def solve_system(self, matvec_object, rhs_np, shape):
        rhs_np = np.ascontiguousarray(rhs_np, dtype=np.float64)
        self.matvec_object = matvec_object
        self.shape = shape

        self.A_shell = PETSc.Mat().createPython(self.shape, comm=PETSc.COMM_WORLD)
        self.A_shell.setPythonContext(self)
        self.A_shell.setUp()
        self.ksp.setOperators(self.A_shell)

        b_petsc = PETSc.Vec().createWithArray(rhs_np, comm=PETSc.COMM_WORLD)
        x_petsc = PETSc.Vec().createWithArray(np.zeros_like(rhs_np), comm=PETSc.COMM_WORLD)
        self.ksp.solve(b_petsc, x_petsc)
        sol = np.array(x_petsc.array_r, copy=True)

        b_petsc.destroy()
        x_petsc.destroy()
        return sol

    def destroy(self, _=None):
        if hasattr(self, "ksp") and self.ksp:
            self.ksp.destroy()
            self.ksp = None
        del self.matvec_object
        del self.shape
        self.x_buffer = None


class IneqStatus(Enum):
    """
    Represents the status of an inequality constraint with specific integer values.
    """
    ACTIVE = 0           # Constraint is active (e.g., g(x) = 0)
    SETTING_ACTIVE = 1   # Constraint is in the process of becoming active
    SETTING_INACTIVE = 2 # Constraint is in the process of becoming inactive
    INACTIVE = 3         # Constraint is inactive (e.g., g(x) < 0)
    NOT_IN_USE = 4

    def __str__(self):
        return self.name.lower().replace('_', ' ')

def _regularized_cholesky(matrix, lower=True):
    try:
        return scp.linalg.cholesky(matrix, check_finite=False, lower=lower, overwrite_a=True)
    except (scp.linalg.LinAlgError, np.linalg.LinAlgError):
        pass

    sym_matrix = 0.5 * (matrix + matrix.T)
    diag_scale = max(float(np.max(np.abs(np.diag(sym_matrix)))), 1.0)
    last_error = None
    for jitter in (1e-12, 1e-10, 1e-8):
        shifted = sym_matrix.copy()
        shifted.flat[::shifted.shape[0] + 1] += jitter * diag_scale
        try:
            return scp.linalg.cholesky(shifted, check_finite=False, lower=lower, overwrite_a=True)
        except (scp.linalg.LinAlgError, np.linalg.LinAlgError) as exc:
            last_error = exc
    raise last_error


def forward_backward_sub(L, b, overwrite_b=False):
    y = scp.linalg.solve_triangular(L, b, lower=True, check_finite=False, overwrite_b=overwrite_b)
    x = scp.linalg.solve_triangular(L.T, y, lower=False, check_finite=False, overwrite_b=True)
    return x

def _ipm_local_solver(XAX_k, block_A_k, XAX_k1, Xb_k, block_b_k, Xb_k1, previous_solution, size_limit, dense_solve=True, rtol=1e-5):
    x_shape = previous_solution.shape
    m = x_shape[0] * x_shape[2] * x_shape[3]
    rhs = np.empty_like(previous_solution)
    rhs[:, 0] = rhs_contract(Xb_k[0], block_b_k[0], Xb_k1[0]) if 0 in block_b_k else 0
    rhs[:, 1] = rhs_contract(Xb_k[1], block_b_k[1], Xb_k1[1]) if 1 in block_b_k else 0
    rhs[:, 2] = rhs_contract(Xb_k[2], block_b_k[2], Xb_k1[2]) if 2 in block_b_k else 0
    norm_rhs = max(np.linalg.norm(rhs), 1e-10)
    inv_I = np.divide(1, core_sum_contract(XAX_k[1, 2], block_A_k[1, 2], XAX_k1[1, 2]))
    block_res_old = np.linalg.norm(block_A_k.block_local_product(XAX_k, XAX_k1, previous_solution).__isub__(rhs)) / norm_rhs
    dense_solve = (np.sqrt(x_shape[0]*x_shape[3]) <= (4.0 / 3.0) * size_limit) and dense_solve and (block_res_old >= rtol)
    direct_solve_failure = not dense_solve

    if dense_solve:
        try:
            mR_p = rhs[:, 0].reshape(m, 1)
            mR_d = rhs[:, 1].reshape(m, 1)
            mR_c = rhs[:, 2].reshape(m, 1)
            L_X_I_inv = dense_core_matrix(XAX_k[2, 2], block_A_k[2, 2], XAX_k1[2, 2])
            L_X_I_inv *= inv_I.reshape(1, -1)
            mL_eq = dense_core_matrix(XAX_k[0, 1], block_A_k[0, 1], XAX_k1[0, 1])
            L_L_Z = _regularized_cholesky(
                dense_core_matrix(XAX_k[2, 1], block_A_k[2, 1], XAX_k1[2, 1]),
                lower=True
            )
            rhs_x = forward_backward_sub(L_L_Z, mR_c - L_X_I_inv @ mR_d, overwrite_b=True)
            A_base = forward_backward_sub(L_L_Z, L_X_I_inv, overwrite_b=True)
            A, b = dense_schur_3block(
                dense_core_matrix(XAX_k[0, 0], block_A_k[0, 0], XAX_k1[0, 0]),
                mL_eq,
                mL_eq.T,
                A_base,
                mR_p,
                rhs_x,
            )
            A.flat[::A.shape[1] + 1] += 1e-11
            solution_now = np.empty(x_shape)
            solution_now[:, 0] = scp.linalg.solve(A, b, check_finite=False, overwrite_a=True, overwrite_b=True, assume_a="gen").reshape(x_shape[0], x_shape[2], x_shape[3])
            solution_now[:, 2] = (
                mR_d - core_rmatvec(XAX_k[0, 1], block_A_k[0, 1], XAX_k1[0, 1], solution_now[:, 0]).reshape(-1, 1)
                ).__imul__(inv_I.reshape(-1, 1)).reshape(x_shape[0], x_shape[2], x_shape[3])
            solution_now[:, 1] = forward_backward_sub(
                L_L_Z,
                mR_c - core_matvec(XAX_k[2, 2], block_A_k[2, 2], XAX_k1[2, 2], solution_now[:, 2]).reshape(-1, 1),
                overwrite_b=True
                ).reshape(x_shape[0], x_shape[2], x_shape[3])
        except (scp.linalg.LinAlgError, np.linalg.LinAlgError):
            direct_solve_failure = True
        except Exception as e:
            tb = traceback.extract_tb(e.__traceback__)
            last = tb[-1]
            print(f"\t⚠️ {type(e).__name__} in {last.filename}, \n\tline {last.lineno}: {last.line.strip()}")
            direct_solve_failure = True

    if not dense_solve or direct_solve_failure:
        matvec_wrapper = MatVecWrapper(
            XAX_k[0, 0], XAX_k[0, 1], XAX_k[2, 1], XAX_k[2, 2],
            block_A_k[0, 0], block_A_k[0, 1], block_A_k[2, 1], block_A_k[2, 2],
            XAX_k1[0, 0], XAX_k1[0, 1], XAX_k1[2, 1], XAX_k1[2, 2],
            inv_I, x_shape[0], x_shape[2], x_shape[3]
        )
        local_rhs = np.empty((2, x_shape[0], x_shape[2], x_shape[3]))
        local_rhs[0] = rhs[:, 0]
        local_rhs[1] = rhs[:, 2]
        local_rhs[1] -= core_matvec(XAX_k[2, 2], block_A_k[2, 2], XAX_k1[2, 2], inv_I*rhs[:, 1])
        local_rhs_norm = np.linalg.norm(local_rhs)
        local_vec = matvec_wrapper.matvec(np.transpose(previous_solution[:, :2], (1, 0, 2, 3)).flatten()).reshape(2, x_shape[0], x_shape[2], x_shape[3])
        local_rhs_norm_prime = np.linalg.norm(local_rhs - local_vec)
        use_prev_sol = (local_rhs_norm_prime < local_rhs_norm)
        if use_prev_sol:
            local_rhs -= local_vec

        num_iters = min(m, 100)
        outer_k = max(num_iters // 10, 3)
        large_scale_solver = LGMRESSolver(rtol=rtol, restart=num_iters, outer_k=outer_k)
        iterative_failure = False
        try:
            local_solution = large_scale_solver.solve_system(matvec_wrapper, local_rhs.flatten(), (2*m, 2*m))
        except Exception as e:
            tb = traceback.extract_tb(e.__traceback__)
            last = tb[-1] if tb else None
            if last is None:
                print(f"	⚠️ {type(e).__name__}: {e}")
            else:
                print(f"\t⚠️ {type(e).__name__} in {last.filename},\n\tline {last.lineno}: {last.line.strip()}")
            iterative_failure = True
            direct_solve_failure = True
            solution_now = previous_solution
        finally:
            large_scale_solver.destroy()

        if not iterative_failure:
            solution_now = np.transpose(local_solution.reshape(2, x_shape[0], x_shape[2], x_shape[3]), (1, 0, 2, 3))

            if use_prev_sol:
                solution_now[:, :2] += previous_solution[:, :2]

            z = inv_I * (rhs[:, 1] - core_rmatvec(XAX_k[0, 1], block_A_k[0, 1], XAX_k1[0, 1], solution_now[:, 0]))
            solution_now = np.concatenate((solution_now, z.reshape(x_shape[0], 1, x_shape[2], x_shape[3])), axis=1)

    block_res_new = np.linalg.norm(block_A_k.block_local_product(XAX_k, XAX_k1, solution_now).__isub__(rhs)) / norm_rhs

    if block_res_old < block_res_new:
        solution_now = previous_solution

    return solution_now, block_res_old, min(block_res_old, block_res_new), rhs, norm_rhs, direct_solve_failure

def _ipm_local_solver_ty(XAX_k, block_A_k, XAX_k1, Xb_k, block_b_k, Xb_k1, previous_solution, size_limit, dense_solve=True, rtol=1e-5):
    x_shape = previous_solution.shape
    m = x_shape[0] * x_shape[2] * x_shape[3]
    rhs = np.empty_like(previous_solution)
    rhs[:, 0] = rhs_contract(Xb_k[0], block_b_k[0], Xb_k1[0]) if 0 in block_b_k else 0
    rhs[:, 1] = rhs_contract(Xb_k[1], block_b_k[1], Xb_k1[1]) if 1 in block_b_k else 0
    rhs[:, 2] = rhs_contract(Xb_k[2], block_b_k[2], Xb_k1[2]) if 2 in block_b_k else 0
    norm_rhs = max(np.linalg.norm(rhs), 1e-10)
    inv_I = np.divide(1, core_sum_contract(XAX_k[1, 2], block_A_k[1, 2], XAX_k1[1, 2]))
    block_res_old = np.linalg.norm(block_A_k.block_local_product(XAX_k, XAX_k1, previous_solution).__isub__(rhs)) / norm_rhs
    dense_solve = (np.sqrt(x_shape[0]*x_shape[3]) <= (4.0 / 3.0) * size_limit) and dense_solve and (block_res_old >= rtol)
    direct_solve_failure = not dense_solve

    if dense_solve:
        try:
            mR_p = rhs[:, 0].reshape(m, 1)
            mR_d = rhs[:, 1].reshape(m, 1)
            mR_c = rhs[:, 2].reshape(m, 1)
            L_X_I_inv = dense_core_matrix(XAX_k[2, 2], block_A_k[2, 2], XAX_k1[2, 2])
            L_X_I_inv *= inv_I.reshape(1, -1)
            A01 = dense_core_matrix(XAX_k[0, 1], block_A_k[0, 1], XAX_k1[0, 1])
            A10 = dense_core_matrix(XAX_k[1, 0], block_A_k[1, 0], XAX_k1[1, 0])
            L_L_Z = _regularized_cholesky(
                dense_core_matrix(XAX_k[2, 1], block_A_k[2, 1], XAX_k1[2, 1]),
                lower=True
            )
            rhs_x = forward_backward_sub(L_L_Z, mR_c - L_X_I_inv @ mR_d, overwrite_b=True)
            A_base = forward_backward_sub(L_L_Z, L_X_I_inv, overwrite_b=True)
            A, b = dense_schur_3block(
                dense_core_matrix(XAX_k[0, 0], block_A_k[0, 0], XAX_k1[0, 0]),
                A01,
                A10,
                A_base,
                mR_p,
                rhs_x,
            )
            A.flat[::A.shape[1] + 1] += 1e-11
            solution_now = np.empty(x_shape)
            solution_now[:, 0] = scp.linalg.solve(A, b, check_finite=False, overwrite_a=True, overwrite_b=True, assume_a="gen").reshape(x_shape[0], x_shape[2], x_shape[3])
            solution_now[:, 2] = (
                mR_d - core_rmatvec(XAX_k[1, 0], block_A_k[1, 0], XAX_k1[1, 0], solution_now[:, 0]).reshape(-1, 1)
                ).__imul__(inv_I.reshape(-1, 1)).reshape(x_shape[0], x_shape[2], x_shape[3])
            solution_now[:, 1] = forward_backward_sub(
                L_L_Z,
                mR_c - core_matvec(XAX_k[2, 2], block_A_k[2, 2], XAX_k1[2, 2], solution_now[:, 2]).reshape(-1, 1),
                overwrite_b=True
            ).reshape(x_shape[0], x_shape[2], x_shape[3])
        except (scp.linalg.LinAlgError, np.linalg.LinAlgError):
            direct_solve_failure = True
        except Exception as e:
            tb = traceback.extract_tb(e.__traceback__)
            last = tb[-1]
            print(f"\t⚠️ {type(e).__name__} in {last.filename}, \n\tline {last.lineno}: {last.line.strip()}")
            direct_solve_failure = True

    if not dense_solve or direct_solve_failure:
        matvec_wrapper = TYMatVecWrapper(
            XAX_k[0, 0], XAX_k[0, 1], XAX_k[1, 0], XAX_k[2, 1], XAX_k[2, 2],
            block_A_k[0, 0], block_A_k[0, 1], block_A_k[1, 0], block_A_k[2, 1], block_A_k[2, 2],
            XAX_k1[0, 0], XAX_k1[0, 1], XAX_k1[1, 0], XAX_k1[2, 1], XAX_k1[2, 2],
            inv_I, x_shape[0], x_shape[2], x_shape[3]
        )
        local_rhs = np.empty((2, x_shape[0], x_shape[2], x_shape[3]))
        local_rhs[0] = rhs[:, 0]
        local_rhs[1] = rhs[:, 2]
        local_rhs[1] -= core_matvec(XAX_k[2, 2], block_A_k[2, 2], XAX_k1[2, 2], inv_I*rhs[:, 1])
        local_rhs_norm = np.linalg.norm(local_rhs)
        local_vec = matvec_wrapper.matvec(np.transpose(previous_solution[:, :2], (1, 0, 2, 3)).flatten()).reshape(2, x_shape[0], x_shape[2], x_shape[3])
        use_prev_sol = np.linalg.norm(local_rhs - local_vec) < local_rhs_norm
        if use_prev_sol:
            local_rhs -= local_vec

        num_iters = min(m, 100)
        outer_k = max(num_iters // 10, 3)
        large_scale_solver = LGMRESSolver(rtol=rtol, restart=num_iters, outer_k=outer_k)
        iterative_failure = False
        try:
            local_solution = large_scale_solver.solve_system(matvec_wrapper, local_rhs.flatten(), (2*m, 2*m))
        except Exception as e:
            tb = traceback.extract_tb(e.__traceback__)
            last = tb[-1] if tb else None
            if last is None:
                print(f"	⚠️ {type(e).__name__}: {e}")
            else:
                print(f"\t⚠️ {type(e).__name__} in {last.filename},\n\tline {last.lineno}: {last.line.strip()}")
            iterative_failure = True
            direct_solve_failure = True
            solution_now = previous_solution
        finally:
            large_scale_solver.destroy()

        if not iterative_failure:
            solution_now = np.transpose(local_solution.reshape(2, x_shape[0], x_shape[2], x_shape[3]), (1, 0, 2, 3))
            if use_prev_sol:
                solution_now[:, :2] += previous_solution[:, :2]
            z = inv_I * (rhs[:, 1] - core_rmatvec(XAX_k[1, 0], block_A_k[1, 0], XAX_k1[1, 0], solution_now[:, 0]))
            solution_now = np.concatenate((solution_now, z.reshape(x_shape[0], 1, x_shape[2], x_shape[3])), axis=1)

    block_res_new = np.linalg.norm(block_A_k.block_local_product(XAX_k, XAX_k1, solution_now).__isub__(rhs)) / norm_rhs

    if block_res_old < block_res_new:
        solution_now = previous_solution

    return solution_now, block_res_old, min(block_res_old, block_res_new), rhs, norm_rhs, direct_solve_failure

def _ipm_local_solver_ineq(XAX_k, block_A_k, XAX_k1, Xb_k, block_b_k, Xb_k1, previous_solution, size_limit, dense_solve=True, rtol=1e-5):
    x_shape = previous_solution.shape
    m = x_shape[0] * x_shape[2] * x_shape[3]
    rhs = np.empty_like(previous_solution)
    rhs[:, 0] = rhs_contract(Xb_k[0], block_b_k[0], Xb_k1[0]) if 0 in block_b_k else 0
    rhs[:, 1] = rhs_contract(Xb_k[1], block_b_k[1], Xb_k1[1]) if 1 in block_b_k else 0
    rhs[:, 2] = rhs_contract(Xb_k[2], block_b_k[2], Xb_k1[2]) if 2 in block_b_k else 0
    rhs[:, 3] = rhs_contract(Xb_k[3], block_b_k[3], Xb_k1[3]) if 3 in block_b_k else 0
    inv_I = np.divide(1, core_sum_contract(XAX_k[1, 2], block_A_k[1, 2], XAX_k1[1, 2]))
    norm_rhs = max(np.linalg.norm(rhs), 1e-10)
    block_res_old = np.linalg.norm(block_A_k.block_local_product(XAX_k, XAX_k1, previous_solution).__isub__(rhs)) / norm_rhs
    dense_solve = (np.sqrt(x_shape[0]*x_shape[3]) <= 0.95*size_limit) and dense_solve and (block_res_old >= rtol)
    direct_solve_failure = not dense_solve

    if dense_solve:
        try:
            L_L_Z = _regularized_cholesky(
                dense_core_matrix(XAX_k[2, 1], block_A_k[2, 1], XAX_k1[2, 1]),
                lower=True
            )
            mR_p = rhs[:, 0].reshape(m, 1)
            mR_d = rhs[:, 1].reshape(m, 1)
            mR_c = rhs[:, 2].reshape(m, 1)
            mR_t = rhs[:, 3].reshape(m, 1)
            L_L_Z_inv_mR_c = forward_backward_sub(L_L_Z, rhs[:, 2].reshape(m, 1))
            L_L_Z_inv_L_X = forward_backward_sub(
                L_L_Z, dense_core_matrix(XAX_k[2, 2], block_A_k[2, 2], XAX_k1[2, 2]), overwrite_b=True
            )
            mL_eq = dense_core_matrix(XAX_k[0, 1], block_A_k[0, 1], XAX_k1[0, 1])
            T_op = dense_core_matrix(XAX_k[3, 1], block_A_k[3, 1], XAX_k1[3, 1])
            A, D, T_op, mL_eq, u, v = dense_schur_4block(
                dense_core_matrix(XAX_k[0, 0], block_A_k[0, 0], XAX_k1[0, 0]),
                dense_core_matrix(XAX_k[3, 3], block_A_k[3, 3], XAX_k1[3, 3]),
                mL_eq,
                T_op,
                L_L_Z_inv_mR_c,
                L_L_Z_inv_L_X,
                inv_I.reshape(-1),
                mR_p,
                mR_d,
                mR_t,
            )
            D.flat[::D.shape[1] + 1] += 1e-11
            Dlu, Dpiv = scp.linalg.lu_factor(D, check_finite=False, overwrite_a=True)
            rhs_l = u.__isub__(mL_eq @ scp.linalg.lu_solve((Dlu, Dpiv), v, check_finite=False))
            lhs_l = A.__isub__(mL_eq.__imatmul__(scp.linalg.lu_solve((Dlu, Dpiv), T_op, check_finite=False)))
            y = scp.linalg.lu_solve(scp.linalg.lu_factor(lhs_l, check_finite=False, overwrite_a=True), rhs_l, check_finite=False, overwrite_b=True)
            solution_now = np.empty(x_shape)
            solution_now[:, 0] = y.reshape(x_shape[0], x_shape[2], x_shape[3])
            solution_now[:, 3] = scp.linalg.lu_solve((Dlu, Dpiv), v.__isub__(T_op @ y), check_finite=False, overwrite_b=True).reshape(x_shape[0], x_shape[2], x_shape[3])
            solution_now[:, 2] = (
                mR_d - core_rmatvec(XAX_k[0, 1], block_A_k[0, 1], XAX_k1[0, 1], solution_now[:, 0]).reshape(-1, 1)
                ).__imul__(inv_I.reshape(-1, 1)).reshape(x_shape[0], x_shape[2], x_shape[3]).__isub__(solution_now[:, 3])
            solution_now[:, 1] = forward_backward_sub(
                L_L_Z,
                mR_c - core_matvec(XAX_k[2, 2], block_A_k[2, 2], XAX_k1[2, 2], solution_now[:, 2]).reshape(-1, 1),
                overwrite_b=True
                ).reshape(x_shape[0], x_shape[2], x_shape[3])

        except (scp.linalg.LinAlgError, np.linalg.LinAlgError):
            direct_solve_failure = True
        except Exception as e:
            tb = traceback.extract_tb(e.__traceback__)
            last = tb[-1]
            print(f"\t⚠️ {type(e).__name__} in {last.filename},\n\tline {last.lineno}: {last.line.strip()}")
            direct_solve_failure = True

    if not dense_solve or direct_solve_failure:

        matvec_wrapper = IneqMatVecWrapper(
            XAX_k[0, 0], XAX_k[0, 1], XAX_k[2, 1], XAX_k[2, 2], XAX_k[3, 1], XAX_k[3, 3],
            block_A_k[0, 0], block_A_k[0, 1], block_A_k[2, 1], block_A_k[2, 2], block_A_k[3, 1], block_A_k[3, 3],
            XAX_k1[0, 0], XAX_k1[0, 1], XAX_k1[2, 1], XAX_k1[2, 2], XAX_k1[3, 1], XAX_k1[3, 3],
            inv_I, x_shape[0], x_shape[2], x_shape[3]
        )
        local_rhs = np.empty((3, x_shape[0], x_shape[2], x_shape[3]))
        local_rhs[0] = rhs[:, 0]
        local_rhs[1] = rhs[:, 2] - core_matvec(
            XAX_k[2, 2], block_A_k[2, 2], XAX_k1[2, 2], inv_I * rhs[:, 1]
        )
        local_rhs[2] = rhs[:, 3]
        local_rhs_norm = np.linalg.norm(local_rhs)
        local_vec = matvec_wrapper.matvec(np.transpose(previous_solution[:, [0, 1, 3]], (1, 0, 2, 3)).flatten()).reshape(3, x_shape[0], x_shape[2], x_shape[3])
        local_rhs_norm_prime = np.linalg.norm(local_rhs - local_vec)
        use_prev_sol = (local_rhs_norm_prime < local_rhs_norm)
        if use_prev_sol:
            local_rhs -= local_vec

        num_iters = min(m, 100)
        outer_k = max(num_iters // 10, 3)
        large_scale_solver = LGMRESSolver(rtol=rtol, restart=num_iters, outer_k=outer_k)
        iterative_failure = False
        try:
            local_solution = large_scale_solver.solve_system(matvec_wrapper, local_rhs.flatten(), (3*m, 3*m))
        except Exception as e:
            tb = traceback.extract_tb(e.__traceback__)
            last = tb[-1] if tb else None
            if last is None:
                print(f"	⚠️ {type(e).__name__}: {e}")
            else:
                print(f"\t⚠️ {type(e).__name__} in {last.filename},\n\tline {last.lineno}: {last.line.strip()}")
            iterative_failure = True
            direct_solve_failure = True
            solution_now = previous_solution
        finally:
            large_scale_solver.destroy()

        if not iterative_failure:
            solution_now = np.transpose(local_solution.reshape(3, x_shape[0], x_shape[2], x_shape[3]),
                                        (1, 0, 2, 3))

            if use_prev_sol:
                solution_now[:, 0] += previous_solution[:, 0]
                solution_now[:, 1] += previous_solution[:, 1]
                solution_now[:, 2] += previous_solution[:, 3]

            z = inv_I * (
                        rhs[:, 1] - core_rmatvec(XAX_k[0, 1], block_A_k[0, 1], XAX_k1[0, 1],
                                                 solution_now[:, 0])) - solution_now[:, 2]
            solution_now = np.concatenate(
                (solution_now[:, :2], z.reshape(x_shape[0], 1, x_shape[2], x_shape[3]), solution_now[:, None, 2]), axis=1)

    block_res_new = np.linalg.norm(block_A_k.block_local_product(XAX_k, XAX_k1, solution_now) - rhs) / norm_rhs

    if block_res_old < block_res_new:
        solution_now = previous_solution

    return solution_now, block_res_old, min(block_res_old, block_res_new), rhs, norm_rhs, direct_solve_failure



@dataclass
class RoundingController:
    residual_factor: float = 0.01
    operator_factor: float = 0.1
    update_factor: float = 0.1
    update_budget_growth: float = 1.25

    def _scaled(self, factor, status, normalisation):
        return factor * status.eta * normalisation

    def residual_primal(self, status):
        return self._scaled(self.residual_factor, status, status.primal_error_normalisation)

    def residual_dual(self, status):
        return self._scaled(self.residual_factor, status, status.dual_error_normalisation)

    def residual_centrality(self, status):
        return self._scaled(self.residual_factor, status, status.centrl_error_normalisation)

    def operator_primal(self, status):
        return self._scaled(self.operator_factor, status, status.primal_error_normalisation)

    def operator_dual(self, status):
        return self._scaled(self.operator_factor, status, status.dual_error_normalisation)

    def operator_centrality(self, status):
        return self._scaled(self.operator_factor, status, status.centrl_error_normalisation)

    def update_primal_base(self, status):
        return self._scaled(self.update_factor, status, status.primal_error_normalisation)

    def update_dual_base(self, status):
        return self._scaled(self.update_factor, status, status.dual_error_normalisation)

    def psd_shift_norm(self, status):
        return np.sqrt(2 ** status.dim)

    def mask_shift_norm(self, status):
        return np.sqrt(max(status.num_ineq_constraints, 1.0))

    def budgeted_round_tol(self, base_tol, shift_norm):
        total_budget = self.update_budget_growth * base_tol * (1.0 + shift_norm)
        return total_budget / (1.0 + shift_norm)

    def update_x_round_tol(self, status):
        return self.budgeted_round_tol(self.update_primal_base(status), self.psd_shift_norm(status))

    def update_z_round_tol(self, status):
        return self.budgeted_round_tol(self.update_dual_base(status), self.psd_shift_norm(status))

    def update_t_round_tol(self, status):
        return self.budgeted_round_tol(self.update_dual_base(status), self.mask_shift_norm(status))


def _tt_rank_sum(train_tt):
    return sum(tt_ranks(train_tt))


def _tt_budgeted_psd_symmetrise(matrix_tt, err_bound):
    matrix_tt = tt_scale(0.5, tt_add(matrix_tt, tt_transpose(matrix_tt)))
    old_rank_sum = _tt_rank_sum(matrix_tt)
    rounded_tt = tt_psd_rank_reduce([core.copy() for core in matrix_tt], eps=err_bound)
    if _tt_rank_sum(rounded_tt) >= old_rank_sum:
        return matrix_tt
    return rounded_tt


def _tt_budgeted_mask_symmetrise(matrix_tt, mask_tt, err_bound):
    matrix_tt = tt_scale(0.5, tt_add(matrix_tt, tt_transpose(matrix_tt)))
    old_rank_sum = _tt_rank_sum(matrix_tt)
    rounded_tt = tt_mask_rank_reduce([core.copy() for core in matrix_tt], mask_tt, eps=err_bound)
    if _tt_rank_sum(rounded_tt) >= old_rank_sum:
        return matrix_tt
    return rounded_tt

def tt_compute_primal_feasibility(lin_op_tt, bias_tt, X_tt, status):
    primal_feas = tt_rank_reduce(tt_sub(tt_mat_vec_mul(lin_op_tt, tt_reshape(X_tt, (4,)), status.rounding.residual_primal(status), status.eps), bias_tt),
                   status.rounding.residual_primal(status))  # primal feasibility
    return primal_feas


def tt_compute_dual_feasibility(obj_tt, lin_op_tt_adj, Z_tt, Y_tt, T_tt, status):
    dual_feas = tt_rank_reduce(tt_sub(tt_fast_matrix_vec_mul(lin_op_tt_adj, Y_tt, status.eps),
                                      tt_rank_reduce(tt_add(tt_reshape(Z_tt, (4,)), obj_tt), status.eps)),
                               status.eps if status.ineq_status is IneqStatus.ACTIVE else status.rounding.residual_dual(status))
    if status.ineq_status is IneqStatus.ACTIVE and T_tt is not None:
        dual_feas = tt_rank_reduce(tt_sub(dual_feas, tt_reshape(T_tt, (4,))), status.rounding.residual_dual(status))

    return dual_feas


def tt_compute_centrality(X_tt, Z_tt, status):
    if status.aho_direction:
        centrality_feas = tt_reshape(tt_scale(-1, _tt_symmetrise(tt_mat_mat_mul(X_tt, Z_tt, status.rounding.residual_centrality(status), status.eps),
                                                        status.rounding.residual_centrality(status))), (4,))
    else:
        centrality_feas = tt_reshape(tt_scale(-1, tt_mat_mat_mul(Z_tt, X_tt, status.rounding.residual_centrality(status), status.eps)), (4,))
    return centrality_feas


def _tt_should_restore_primal(status):
    if status.is_last_iter or status.is_primal_feasible or not status.is_dual_feasible:
        return False
    reference_error = max(status.dual_error, status.centrality_error, status.feasibility_tol)
    return status.primal_error > 10.0 * reference_error


def tt_infeasible_newton_system(
        lhs,
        obj_tt,
        X_tt,
        Y_tt,
        Z_tt,
        T_tt,
        lin_op_tt,
        lin_op_tt_adj,
        bias_tt,
        ineq_mask,
        status
):
    rhs = TTBlockVector()
    primal_feas = tt_compute_primal_feasibility(lin_op_tt, bias_tt, X_tt, status)
    status.primal_feas_norm = tt_norm(primal_feas)
    status.primal_error = np.divide(status.primal_feas_norm, status.primal_error_normalisation)
    status.is_primal_feasible = np.less(status.primal_error, status.feasibility_tol)

    dual_feas = tt_compute_dual_feasibility(obj_tt, lin_op_tt_adj, Z_tt, Y_tt, T_tt, status)
    status.dual_feas_norm = tt_norm(dual_feas)
    status.dual_error = np.divide(status.dual_feas_norm, status.dual_error_normalisation)
    status.is_dual_feasible = np.less(status.dual_error, (1 + (status.ineq_status is IneqStatus.ACTIVE))*status.feasibility_tol)

    status.is_last_iter = status.is_last_iter or (status.is_primal_feasible and status.is_dual_feasible and status.is_central)
    wants_primal_restoration = _tt_should_restore_primal(status)
    if wants_primal_restoration and status.primal_restoration_steps < status.max_primal_restoration_steps:
        status.primal_restoration = True
        status.primal_restoration_steps += 1
    else:
        status.primal_restoration = False
        if not wants_primal_restoration or status.primal_restoration_steps >= status.max_primal_restoration_steps:
            status.primal_restoration_steps = 0
    if status.primal_restoration:
        status.aho_direction = False

    if status.aho_direction:
        lhs[2, 1] = tt_psd_rank_reduce(tt_scale(0.5, tt_add(tt_IkronM(Z_tt), tt_MkronI(Z_tt))), eps=status.rounding.operator_dual(status))
        lhs[2, 2] = tt_psd_rank_reduce(tt_scale(0.5, tt_add(tt_MkronI(X_tt), tt_IkronM(X_tt))), eps=status.rounding.operator_primal(status))
    else:
        lhs[2, 1] = tt_psd_rank_reduce(tt_MkronI(Z_tt), eps=status.rounding.operator_dual(status))
        lhs[2, 2] = tt_psd_rank_reduce(tt_IkronM(X_tt), eps=status.rounding.operator_primal(status))

    needs_primal_row = not status.is_primal_feasible or status.is_last_iter
    needs_dual_row = (not status.is_dual_feasible or status.is_last_iter) and not status.primal_restoration
    needs_central_row = not status.is_central or status.is_last_iter

    if getattr(status, "combine_ty", False) and status.ineq_status is IneqStatus.ACTIVE:
        masked_X_tt = tt_rank_reduce(
            tt_add(tt_scale(status.ineq_boundary_val, ineq_mask), tt_fast_hadamard(ineq_mask, X_tt, status.eps)),
            eps=status.eps
        )
        lhs[0, 0] = tt_rank_reduce(
            tt_add(status.lag_map_ty, tt_diag_op(masked_X_tt, status.eps)),
            eps=status.rounding.operator_dual(status)
        )
        lhs[0, 1] = tt_rank_reduce(
            tt_add(tt_scale(-1, lin_op_tt), tt_diag_op(T_tt, status.rounding.operator_dual(status))),
            eps=status.rounding.operator_dual(status)
        )
        lhs[1, 0] = tt_rank_reduce(
            tt_add(tt_scale(-1, lin_op_tt_adj), tt_diag_op(ineq_mask, status.eps)),
            eps=status.rounding.operator_dual(status)
        )

        if needs_primal_row:
            rhs[0] = primal_feas
        if needs_central_row:
            ineq_rhs = tt_rank_reduce(
                tt_reshape(tt_scale(-1, tt_fast_hadamard(masked_X_tt, T_tt, status.eps)), (4, )),
                eps=status.rounding.residual_centrality(status)
            )
            rhs[0] = tt_rank_reduce(
                tt_add(rhs.get_row(0), ineq_rhs) if rhs.get_row(0) is not None else ineq_rhs,
                eps=status.rounding.residual_centrality(status)
            )
    else:
        if needs_primal_row:
            rhs[0] = primal_feas

    if needs_dual_row:
        rhs[1] = dual_feas

    if needs_central_row:
        rhs[2] = tt_compute_centrality(X_tt, Z_tt, status)

    if status.ineq_status is IneqStatus.ACTIVE and not getattr(status, "combine_ty", False):
        lhs[3, 1] =  tt_diag_op(T_tt, status.rounding.operator_dual(status))
        masked_X_tt = tt_rank_reduce(tt_add(tt_scale(status.ineq_boundary_val, ineq_mask), tt_fast_hadamard(ineq_mask, X_tt, status.eps)), eps=status.eps)
        lhs[3, 3] = tt_rank_reduce(tt_add(status.lag_map_t, tt_diag_op(masked_X_tt, status.eps)), eps=status.rounding.operator_dual(status))
        if needs_central_row:
            rhs[3] = tt_rank_reduce(tt_reshape(tt_scale(-1, tt_fast_hadamard(masked_X_tt, T_tt, status.eps)), (4, )), eps=status.rounding.residual_centrality(status))
    return lhs, rhs, status

def _tt_symmetrise(matrix_tt, err_bound):
    return tt_rank_reduce(tt_scale(0.5, tt_add(matrix_tt, tt_transpose(matrix_tt))), eps=err_bound)

def _tt_psd_symmetrise(matrix_tt, err_bound):
    return tt_psd_rank_reduce(tt_scale(0.5, tt_add(matrix_tt, tt_transpose(matrix_tt))), eps=err_bound)


def _tt_mask_symmetrise(matrix_tt, mask_tt, err_bound):
    return tt_mask_rank_reduce(tt_scale(0.5, tt_add(matrix_tt, tt_transpose(matrix_tt))), mask_tt, eps=err_bound)

def _tt_get_block(i, block_matrix_tt):
    b = np.argmax([len(c.shape) for c in block_matrix_tt])
    return block_matrix_tt[:b] + [block_matrix_tt[b][:, i]] + block_matrix_tt[b+1:]


def _tt_copy(train_tt):
    return [np.array(core, copy=True) for core in train_tt]


def _tt_scale_nondestructive(train_tt, scale):
    if train_tt is None or np.isclose(scale, 1.0):
        return train_tt
    return tt_scale(scale, _tt_copy(train_tt))


def _tt_rhs_row_norm(rhs_vec_tt, row_index):
    rhs_row = rhs_vec_tt.get_row(row_index)
    if rhs_row is None:
        return 0.0
    row_norm = tt_norm(rhs_row)
    return float(row_norm) if np.isfinite(row_norm) else 0.0


def _tt_cached_rhs_row_norm(rhs_vec_tt, row_index, cached_norm):
    if rhs_vec_tt.get_row(row_index) is None:
        return 0.0
    if cached_norm is not None and np.isfinite(cached_norm):
        return float(cached_norm)
    return _tt_rhs_row_norm(rhs_vec_tt, row_index)


def _tt_kkt_row_scales(rhs_vec_tt, status):
    eps = max(status.op_tol, 1e-12)
    feas_norm = max(
        _tt_cached_rhs_row_norm(rhs_vec_tt, 0, getattr(status, "primal_feas_norm", None)),
        _tt_cached_rhs_row_norm(rhs_vec_tt, 1, getattr(status, "dual_feas_norm", None)),
    )
    cent_norm = max(_tt_rhs_row_norm(rhs_vec_tt, 2), _tt_rhs_row_norm(rhs_vec_tt, 3))

    row_scales = {}
    if feas_norm > eps:
        feas_scale = float(np.clip(1.0 / max(feas_norm, eps), 1e-6, 1e6))
        row_scales[0] = feas_scale
        row_scales[1] = feas_scale
    if cent_norm > eps:
        cent_scale = float(np.clip(1.0 / max(cent_norm, eps), 1e-6, 1e6))
        if 0 in row_scales:
            # Keep feasibility equations at least as important as centrality.
            cent_scale = min(cent_scale, row_scales[0])
        row_scales[2] = cent_scale
        row_scales[3] = cent_scale

    return row_scales


def _tt_effective_row_scale(lhs_matrix_tt, key, row_scales):
    row_index = key[0]
    scale = row_scales.get(row_index, 1.0)
    if key in lhs_matrix_tt._transposes:
        coupled_row, _ = lhs_matrix_tt._transposes[key]
        if coupled_row in row_scales:
            scale = np.sqrt(scale * row_scales[coupled_row])
    if key in lhs_matrix_tt._aliases:
        coupled_row, _ = lhs_matrix_tt._aliases[key]
        if coupled_row in row_scales:
            scale = np.sqrt(scale * row_scales[coupled_row])
    return float(scale)


def _tt_scale_kkt_rhs(rhs_vec_tt, row_scales):
    if not row_scales:
        return rhs_vec_tt
    rhs_scaled = TTBlockVector()
    for row_index in rhs_vec_tt.keys():
        scale = row_scales.get(row_index, 1.0)
        rhs_scaled[row_index] = _tt_scale_nondestructive(rhs_vec_tt.get_row(row_index), scale)
    return rhs_scaled


def _tt_build_row_scaled_kkt(lhs_matrix_tt, rhs_vec_tt, status, row_scales=None):
    if row_scales is None:
        row_scales = _tt_kkt_row_scales(rhs_vec_tt, status)
    if not row_scales:
        return lhs_matrix_tt, rhs_vec_tt

    lhs_scaled = TTBlockMatrix()
    lhs_scaled._aliases = dict(lhs_matrix_tt._aliases)
    lhs_scaled._transposes = dict(lhs_matrix_tt._transposes)
    for key, block in lhs_matrix_tt._data.items():
        scale = _tt_effective_row_scale(lhs_matrix_tt, key, row_scales)
        lhs_scaled[key] = _tt_scale_nondestructive(block, scale)

    rhs_scaled = _tt_scale_kkt_rhs(rhs_vec_tt, row_scales)

    if status.verbose:
        feas_scale = row_scales.get(0, row_scales.get(1, 1.0))
        cent_scale = row_scales.get(2, row_scales.get(3, 1.0))
        print(f"KKT row scaling: feas={feas_scale:.2e}, cent={cent_scale:.2e}", flush=True)

    return lhs_scaled, rhs_scaled


def _tt_amen_fallback_triggered(exc):
    message = str(exc)
    return isinstance(exc, RuntimeError) and (
        "Target residual not reached" in message
        or "Number of restarts exhausted" in message
    )


def _tt_clone_block_matrix(block_matrix_tt):
    clone = TTBlockMatrix()
    clone._data = dict(block_matrix_tt._data)
    clone._aliases = dict(block_matrix_tt._aliases)
    clone._transposes = dict(block_matrix_tt._transposes)
    return clone


def _tt_clone_block_vector(block_vec_tt):
    clone = TTBlockVector()
    for row_index in block_vec_tt.keys():
        clone[row_index] = block_vec_tt.get_row(row_index)
    return clone


def _tt_xz_fallback_kkt(lhs_matrix_tt, rhs_vec_tt, X_tt, Z_tt, status):
    if status.verbose:
        print("\tAHO KKT solve missed target residual; retrying predictor with XZ direction.", flush=True)
    status.aho_direction = False
    status.mals_delta0 = None
    lhs_xz = _tt_clone_block_matrix(lhs_matrix_tt)
    lhs_xz[2, 1] = tt_psd_rank_reduce(tt_MkronI(Z_tt), eps=status.rounding.operator_dual(status))
    lhs_xz[2, 2] = tt_psd_rank_reduce(tt_IkronM(X_tt), eps=status.rounding.operator_primal(status))
    rhs_xz = _tt_clone_block_vector(rhs_vec_tt)
    if rhs_vec_tt.get_row(2) is not None:
        rhs_xz[2] = tt_compute_centrality(X_tt, Z_tt, status)
    return lhs_xz, rhs_xz


def _tt_split_dual_directions(delta_dual_tt, ineq_mask, status):
    if not getattr(status, "combine_ty", False):
        return tt_rank_reduce(delta_dual_tt, eps=status.eps), None
    delta_dual_mat = tt_reshape(tt_rank_reduce(delta_dual_tt, eps=status.eps), (2, 2))
    delta_y = tt_rank_reduce(
        tt_reshape(tt_fast_hadamard(status.eq_mask, delta_dual_mat, status.eps), (4, )),
        eps=status.eps
    )
    delta_t = tt_fast_hadamard(ineq_mask, delta_dual_mat, status.eps)
    return delta_y, delta_t


def _tt_ipm_newton_step(
        lhs_matrix_tt,
        rhs_vec_tt,
        ineq_mask,
        X_tt,
        Z_tt,
        T_tt,
        ZX,
        TX,
        status,
        solver
):
    try:
        newton_t0 = time.time()
        _ipm_trace(status, "newton", f"start dir={'AHO' if status.aho_direction else 'XZ'} ineq={status.ineq_status.name.lower()} eta={status.eta:.2e}")
        # Predictor
        if status.verbose:
            print("\n--- Predictor  step ---", flush=True)
        t0 = time.time()
        row_scales = _tt_kkt_row_scales(rhs_vec_tt, status)
        scale_log = [f"{row}:{scale:.1e}" for row, scale in row_scales.items()]
        lhs_pred_tt, rhs_pred_tt = _tt_build_row_scaled_kkt(lhs_matrix_tt, rhs_vec_tt, status, row_scales=row_scales)
        _ipm_trace(status, "kkt-build", f"rhs_rows={list(rhs_vec_tt.keys())} scales={scale_log}", t0)
        try:
            t0 = time.time()
            _ipm_trace(status, "kkt-pred", f"start nswp={status.kkt_iterations + status.is_last_iter} rank_cap={status.mals_rank_restriction}")
            Delta_tt, _ = solver(
                lhs_pred_tt,
                rhs_pred_tt,
                status.mals_delta0,
                status.kkt_iterations + status.is_last_iter,
                status.mals_rank_restriction,
                status.eta,
                strict_first_attempt=status.aho_direction,
            )
            _ipm_trace(status, "kkt-pred", "done", t0)
        except RuntimeError as solve_exc:
            if not (status.aho_direction and _tt_amen_fallback_triggered(solve_exc)):
                raise
            _ipm_trace(status, "kkt-pred", "fallback=aho_to_xz")
            lhs_matrix_tt, rhs_vec_tt = _tt_xz_fallback_kkt(lhs_matrix_tt, rhs_vec_tt, X_tt, Z_tt, status)
            t0 = time.time()
            row_scales = _tt_kkt_row_scales(rhs_vec_tt, status)
            lhs_pred_tt, rhs_pred_tt = _tt_build_row_scaled_kkt(lhs_matrix_tt, rhs_vec_tt, status, row_scales=row_scales)
            _ipm_trace(status, "kkt-build", f"fallback rhs_rows={list(rhs_vec_tt.keys())}", t0)
            t0 = time.time()
            Delta_tt, _ = solver(lhs_pred_tt, rhs_pred_tt, status.mals_delta0, status.kkt_iterations + status.is_last_iter, status.mals_rank_restriction, status.eta)
            _ipm_trace(status, "kkt-pred", "fallback done", t0)
        status.mals_delta0 = Delta_tt
        t0 = time.time()
        Delta_X_tt = _tt_symmetrise(tt_reshape(_tt_get_block(1, Delta_tt), (2, 2)), status.eps)
        Delta_Z_tt = _tt_symmetrise(tt_reshape(_tt_get_block(2, Delta_tt), (2, 2)), status.eps)
        Delta_Y_tt, Delta_T_tt = _tt_split_dual_directions(_tt_get_block(0, Delta_tt), ineq_mask, status)
        if status.ineq_status is IneqStatus.ACTIVE and not getattr(status, "combine_ty", False):
            Delta_T_tt = tt_rank_reduce(_tt_get_block(3, Delta_tt), eps=status.eps)
            Delta_T_tt = tt_fast_hadamard(ineq_mask, tt_reshape(Delta_T_tt, (2, 2)), status.eps)
        _ipm_trace(status, "delta", f"rmax dX/dY/dZ/dT={_tt_rank_peak(Delta_X_tt)}/{_tt_rank_peak(Delta_Y_tt)}/{_tt_rank_peak(Delta_Z_tt)}/{_tt_rank_peak(Delta_T_tt)}", t0)

        t0 = time.time()
        _ipm_trace(status, "step-size", "start")
        x_step_size, z_step_size = _tt_get_step_sizes(
            X_tt,
            Z_tt,
            T_tt,
            Delta_X_tt,
            Delta_Z_tt,
            Delta_T_tt,
            ineq_mask,
            status
        )
        _ipm_trace(status, "step-size", f"alpha_x={x_step_size:.2e} alpha_z={z_step_size:.2e}", t0)

        if status.primal_restoration:
            x_step_size *= 0.5
            z_step_size *= 0.5

        if not status.primal_restoration and not status.is_central and not status.is_last_iter:

            DXZ = tt_inner_prod(Delta_X_tt, Delta_Z_tt)
            corrector_needed = False
            # Corrector
            if status.verbose:
                print(f"\n--- Centering-Corrector  step ---", flush=True)

            if status.ineq_status is IneqStatus.ACTIVE:
                mu_aff = (
                    ZX + x_step_size * z_step_size * DXZ
                    + z_step_size * tt_inner_prod(X_tt, Delta_Z_tt)
                    + x_step_size * tt_inner_prod(Delta_X_tt, Z_tt)
                    + TX
                    + x_step_size * tt_inner_prod(Delta_X_tt, T_tt)
                )
                mu_aff += (
                    x_step_size * z_step_size * tt_inner_prod(Delta_T_tt, Delta_X_tt)
                    + z_step_size * (tt_inner_prod(X_tt, Delta_T_tt) + status.ineq_boundary_val*tt_entrywise_sum(Delta_T_tt))
                )
                e = max(1, 3 * min(x_step_size, z_step_size) ** 2)
                status.sigma = min(0.99, max(mu_aff/(ZX + TX), 0)**e)
                if status.sigma > 1e-4:
                    sigma_rhs = tt_scale(status.sigma * status.mu, tt_reshape(ineq_mask, (4,)))
                    row = 0 if getattr(status, "combine_ty", False) else 3
                    rhs_vec_tt[row] = tt_rank_reduce(tt_add(
                            sigma_rhs,
                            rhs_vec_tt.get_row(row),
                            ), status.rounding.operator_centrality(status)
                            )
                    corrector_needed = True
            else:
                mu_aff = (
                    ZX + x_step_size * z_step_size * DXZ
                    + z_step_size * tt_inner_prod(X_tt,Delta_Z_tt)
                    + x_step_size * tt_inner_prod(Delta_X_tt, Z_tt)
                )
                e = max(1, 3*min(x_step_size, z_step_size)**2)
                status.sigma = min(0.99, max(mu_aff/ZX, 0) ** e)


            if DXZ > 0.1*status.centrality_tol:
                Delta_XZ_term = tt_compute_centrality(Delta_X_tt, Delta_Z_tt, status)
                rhs_vec_tt[2] = tt_rank_reduce(
                    tt_add(
                        tt_scale(status.sigma * status.mu, tt_reshape(tt_identity(len(X_tt)), (4,))),
                        tt_add(
                            rhs_vec_tt.get_row(2),
                            Delta_XZ_term
                        )
                    ),
                    status.rounding.operator_centrality(status)
                ) if status.sigma > 1e-4 else tt_rank_reduce(tt_add(rhs_vec_tt.get_row(2), Delta_XZ_term), status.rounding.operator_centrality(status))
                corrector_needed = True
            elif status.sigma > 1e-4:
                rhs_vec_tt[2] = tt_rank_reduce(
                    tt_add(
                        tt_scale(status.sigma * status.mu, tt_reshape(tt_identity(len(X_tt)), (4,))),
                        rhs_vec_tt.get_row(2)
                    ),
                    status.rounding.operator_centrality(status)
                )
                corrector_needed = True

            if corrector_needed:
                _ipm_trace(status, "corrector", f"needed sigma={status.sigma:.2e} dxz={DXZ:.2e}")
                lhs_cc_tt = lhs_pred_tt
                t0 = time.time()
                rhs_cc_tt = _tt_scale_kkt_rhs(rhs_vec_tt, row_scales)
                _ipm_trace(status, "rhs-scale", "corrector", t0)
                t0 = time.time()
                _ipm_trace(status, "kkt-corr", f"start nswp={status.kkt_iterations + status.is_last_iter} rank_cap={status.mals_rank_restriction}")
                Delta_tt_cc, _ = solver(lhs_cc_tt, rhs_cc_tt, status.mals_delta0, status.kkt_iterations + status.is_last_iter, status.mals_rank_restriction, status.eta)
                _ipm_trace(status, "kkt-corr", "done", t0)
                status.mals_delta0 = Delta_tt_cc
                Delta_X_tt_cc = _tt_symmetrise(tt_reshape(_tt_get_block(1, Delta_tt_cc), (2, 2)), status.eps)
                Delta_Z_tt_cc = _tt_symmetrise(tt_reshape(_tt_get_block(2, Delta_tt_cc), (2, 2)), status.eps)
                Delta_Y_tt_cc, Delta_T_tt_cc = _tt_split_dual_directions(_tt_get_block(0, Delta_tt_cc), ineq_mask, status)
                Delta_X_tt = tt_rank_reduce(tt_add(Delta_X_tt_cc, Delta_X_tt), eps=status.eps)
                Delta_Y_tt = tt_rank_reduce(tt_add(Delta_Y_tt_cc, Delta_Y_tt), eps=status.eps)
                Delta_Z_tt = tt_rank_reduce(tt_add(Delta_Z_tt_cc, Delta_Z_tt), eps=status.eps)
                if status.ineq_status is IneqStatus.ACTIVE:
                    if not getattr(status, "combine_ty", False):
                        Delta_T_tt_cc = tt_rank_reduce(_tt_get_block(3, Delta_tt_cc), eps=status.eps)
                        Delta_T_tt_cc = tt_fast_hadamard(ineq_mask, tt_reshape(Delta_T_tt_cc, (2, 2)), status.eps)
                    Delta_T_tt = tt_rank_reduce(tt_add(Delta_T_tt_cc, Delta_T_tt), eps=status.eps)

                t0 = time.time()
                _ipm_trace(status, "step-size", "corrector start")
                x_step_size, z_step_size = _tt_get_step_sizes(
                    X_tt,
                    Z_tt,
                    T_tt,
                    Delta_X_tt,
                    Delta_Z_tt,
                    Delta_T_tt,
                    ineq_mask,
                    status
                )
                _ipm_trace(status, "step-size", f"corrector alpha_x={x_step_size:.2e} alpha_z={z_step_size:.2e}", t0)
            else:
                _ipm_trace(status, "corrector", f"skipped sigma={status.sigma:.2e} dxz={DXZ:.2e}")
        else:
            status.sigma = 0
            _ipm_trace(status, "corrector", "skipped by phase/centrality")
    except Exception as e:
        print(f"\n\t⚠️ Attention: {e}")
        print("\n\t==> Full traceback (most recent call last):")
        traceback.print_exc(file=sys.stdout)
        return 0, 0, None, None, None, None, status

    _ipm_trace(status, "newton", f"done alpha_x={x_step_size:.2e} alpha_z={z_step_size:.2e}", newton_t0)
    return x_step_size, z_step_size, Delta_X_tt, Delta_Y_tt, Delta_Z_tt, Delta_T_tt, status


def _tt_get_step_sizes(
        X_tt,
        Z_tt,
        T_tt,
        Delta_X_tt,
        Delta_Z_tt,
        Delta_T_tt,
        ineq_mask,
        status
):
    if status.is_last_iter:
        X_tt = tt_add(X_tt, tt_scale(status.boundary_val, tt_identity(len(X_tt))))
        Z_tt = tt_add(Z_tt, tt_scale(status.boundary_val, tt_identity(len(Z_tt))))

    t0 = time.time()
    x_step_size, status.eigen_x0 = tt_max_generalised_eigen(X_tt, Delta_X_tt, x0=status.eigen_x0, tol=1e-8, verbose=status.verbose)
    _ipm_trace(status, "step-x", f"alpha={x_step_size:.2e}", t0)
    t0 = time.time()
    z_step_size, status.eigen_z0 = tt_max_generalised_eigen(Z_tt, Delta_Z_tt, x0=status.eigen_z0, tol=1e-8, verbose=status.verbose)
    _ipm_trace(status, "step-z", f"alpha={z_step_size:.2e}", t0)
    if status.ineq_status is not IneqStatus.NOT_IN_USE:
        if status.is_last_iter:
            X_tt = tt_add(X_tt, tt_scale(status.ineq_boundary_val + status.boundary_val, ineq_mask))
            T_tt = tt_add(T_tt, tt_scale(status.ineq_boundary_val + status.boundary_val, ineq_mask))
        t0 = time.time()
        x_step_size, z_step_size = _tt_get_ineq_step_sizes(x_step_size, z_step_size, X_tt, T_tt, Delta_X_tt, Delta_T_tt, ineq_mask, status)
        _ipm_trace(status, "step-ineq", f"alpha_x={x_step_size:.2e} alpha_z={z_step_size:.2e}", t0)
    tau_x = 0.9 + 0.05*min(x_step_size,  z_step_size)
    tau_z = 0.9 + 0.05*min(x_step_size,  z_step_size)

    if status.verbose:
        print(f"Step search concluded.")
        print(f"Step sizes: a_p:{x_step_size:.2e}, a_d:{z_step_size:.2e}", flush=True)
    return tau_x*x_step_size, tau_z*z_step_size


def _dense_masked_ineq_step_size(A_tt, Delta_tt, ineq_mask, status, max_entries=1 << 18):
    if ineq_mask is None:
        return None
    num_entries = int(np.prod([np.prod(core.shape[1:-1], dtype=int) for core in ineq_mask], dtype=int))
    if num_entries > max_entries:
        return None

    mask = np.abs(tt_to_tensor(ineq_mask).reshape(-1)) > status.eps
    if not np.any(mask):
        return 1.0
    A = tt_to_tensor(A_tt).reshape(-1)[mask]
    Delta = tt_to_tensor(Delta_tt).reshape(-1)[mask]
    descending = Delta < -status.eps
    if not np.any(descending):
        return 1.0
    ratios = -A[descending] / Delta[descending]
    ratios = ratios[np.isfinite(ratios)]
    if ratios.size == 0:
        return 1.0
    return float(np.clip(np.min(ratios), a_min=0.0, a_max=1.0))


def _ineq_step_size(A_tt, Delta_tt, e_tt, ineq_mask, status):
    dense_step_size = _dense_masked_ineq_step_size(A_tt, Delta_tt, ineq_mask, status)
    if dense_step_size is not None:
        return dense_step_size, e_tt

    sum_tt = tt_add(A_tt, Delta_tt)
    if status.compl_ineq_mask:
        sum_tt = tt_add(sum_tt, status.compl_ineq_mask)
    sum_tt = tt_rank_reduce(sum_tt, status.eps)
    e_tt, _ = tt_min_eig(tt_diag_op(sum_tt, status.eps), x0=e_tt, tol=1e-8, verbose=status.verbose)
    e_tt_sq = tt_reshape(e_tt, (2, 2))
    if np.abs(tt_inner_prod(sum_tt, e_tt_sq)) > status.eps:
        e_tt_sq = tt_normalise(tt_fast_hadamard(e_tt_sq, e_tt_sq, status.eps))
        min_A_val = np.abs(tt_inner_prod(A_tt, e_tt_sq))
        min_Delta_val = tt_inner_prod(Delta_tt, e_tt_sq)
        if min_Delta_val >= -status.eps:
            step_size = 1
        else:
            step_size = np.clip(-min_A_val/min_Delta_val, a_min=0, a_max=1)
    else: # min is one of the padded entries
        step_size = 1
    return step_size, e_tt


def _tt_get_ineq_step_sizes(x_step_size, z_step_size, X_tt, T_tt, Delta_X_tt, Delta_T_tt, ineq_mask, status):
    status.ineq_next_status = None

    if x_step_size > 0:
        masked_X_tt = tt_fast_hadamard(ineq_mask, X_tt, status.eps)
        masked_Delta_X_tt = tt_fast_hadamard(ineq_mask, Delta_X_tt, status.eps)
        x_ineq_step_size, status.eigen_xt0 = _ineq_step_size(
            tt_add(masked_X_tt, tt_scale(status.ineq_boundary_val, ineq_mask)),
            tt_scale(x_step_size, masked_Delta_X_tt),
            status.eigen_xt0,
            ineq_mask,
            status
        )
        if not status.is_last_iter and not getattr(status, "combine_ty", False):
            step_deficit = 1 - x_ineq_step_size
            if status.ineq_status is IneqStatus.ACTIVE:
                if step_deficit < status.op_tol:
                    status.ineq_full_step_streak += 1
                else:
                    status.ineq_full_step_streak = 0
                primal_dual_ready = (
                    status.primal_error < 10.0*status.feasibility_tol
                    and status.dual_error < 10.0*status.feasibility_tol
                )
                t_small = tt_norm(T_tt) < status.op_tol
                tx_small = status.ineq_slack <= 0.1*max(status.psd_slack, status.op_tol)
                switch_cooldown = status.iteration - status.ineq_last_switch_iter >= 3
                if (primal_dual_ready and t_small and tx_small
                        and switch_cooldown and status.ineq_full_step_streak >= 3):
                    status.ineq_next_status = IneqStatus.INACTIVE
            elif status.ineq_status is IneqStatus.INACTIVE:
                if step_deficit > 10.0*status.op_tol:
                    status.ineq_next_status = IneqStatus.ACTIVE
        x_step_size *= x_ineq_step_size

    if (z_step_size > 0 and status.ineq_status is IneqStatus.ACTIVE
            and Delta_T_tt is not None):
        t_step_size, status.eigen_zt0 = _ineq_step_size(
            T_tt,
            tt_scale(z_step_size, Delta_T_tt),
            status.eigen_zt0,
            ineq_mask,
            status
        )
        z_step_size *= t_step_size

    return x_step_size, z_step_size


def _tt_update_ineq_boundary(status, ZX, raw_TX, T_tt):
    if status.ineq_status is not IneqStatus.ACTIVE or T_tt is None:
        return status.ineq_boundary_val

    t_sum = abs(tt_entrywise_sum(T_tt))
    if not np.isfinite(t_sum) or t_sum <= 0:
        return status.ineq_boundary_val

    denom = 2 ** status.dim + max(status.num_ineq_constraints, 0.0)
    raw_mu = (abs(ZX) + abs(raw_TX)) / max(denom, 1.0)
    target = status.ineq_boundary_fraction * raw_mu * max(status.num_ineq_constraints, 1.0) / max(t_sum, 1e-300)
    target = float(np.clip(target, status.ineq_boundary_min, status.ineq_boundary_max))
    return min(status.ineq_boundary_val, target)


def _initialise(ineq_mask, status, dim, lambdaStar, lambdaStarIneq):
    X_tt = tt_scale(lambdaStar, tt_identity(dim))
    Z_tt = tt_scale(lambdaStar, tt_identity(dim))
    Y_tt = tt_reshape(tt_zero_matrix(dim), (4, ))
    T_tt = None

    if status.ineq_status is IneqStatus.ACTIVE:
        T_tt = tt_scale(lambdaStarIneq, ineq_mask)
        # Need to initialise so it stays psd
        x_step_size, _ = tt_max_generalised_eigen(X_tt, ineq_mask, tol=1e-7, verbose=status.verbose)
        X_tt = tt_rank_reduce(tt_add(X_tt, tt_scale(0.1*x_step_size, ineq_mask)), status.rounding.update_x_round_tol(status))

    return X_tt, Y_tt, Z_tt, T_tt

@dataclass
class IPMStatus:
    dim: int
    feasibility_tol: float
    centrality_tol: float
    op_tol: float
    eps: float

    aho_direction: bool
    is_primal_feasible: bool
    primal_error: float
    is_dual_feasible: bool
    dual_error: float
    is_central: bool
    centrality_error: float
    mu: float

    is_last_iter: bool
    ineq_status: IneqStatus
    verbose: bool

    primal_error_normalisation: float
    dual_error_normalisation: float
    mals_rank_restriction: int

    boundary_val: float = 1e-10
    ineq_boundary_val: float = 0.03
    ineq_boundary_min: float = 0.01
    ineq_boundary_max: float = 0.03
    ineq_boundary_fraction: float = 0.1
    sigma: float = 0.5
    num_ineq_constraints: float = 0
    lag_map_t = None
    lag_map_y = None
    lag_map_ty = None
    eq_mask = None
    ty_idle_mask = None
    compl_ineq_mask = None
    mals_delta0 = None
    eigen_x0 = None
    eigen_z0 = None
    eigen_xt0 = None
    eigen_zt0 = None
    kkt_iterations = 7
    centrl_error_normalisation: float = 1.0
    primal_feas_norm: float = np.inf
    dual_feas_norm: float = np.inf
    eta = 1e-3
    primal_restoration: bool = False
    primal_restoration_steps: int = 0
    max_primal_restoration_steps: int = 2
    trace_verbose: bool = False
    ineq_next_status = None
    ineq_full_step_streak: int = 0
    ineq_last_switch_iter: int = -1000000
    iteration: int = 0
    psd_slack: float = np.inf
    ineq_slack: float = np.inf
    combine_ty: bool = False


def _ipm_format_output(X_tt, Y_tt, T_tt, Z_tt, iteration, status):
    """Formats the final results into the desired output structure."""
    ranksX = tt_ranks(X_tt)
    ranksZ = tt_ranks(Z_tt)
    ranksY = tt_ranks(Y_tt)
    ranksT = tt_ranks(T_tt) if T_tt else [0] * (status.dim - 1)

    print("---Terminated---")
    print(f"Terminated in {iteration} iterations.")
    print(f"Ranks: X={ranksX}, Z={ranksZ}, Y={ranksY}, T={ranksT}")

    results = {"num_iters": iteration, "ranksX": ranksX, "ranksY": ranksY, "ranksZ": ranksZ, "ranksT": ranksT, "status": status}
    return X_tt, Y_tt, T_tt, Z_tt, results


def _ipm_check_for_stalled_progress(prev_errors, status, gap_tol):
    """Checks if the optimization has stalled."""
    if status.is_last_iter:
        return False

    primal_stalled = abs(prev_errors['primal'] - status.primal_error) < 0.04 * gap_tol
    dual_stalled = abs(prev_errors['dual'] - status.dual_error) < 0.04 * gap_tol
    centrality_stalled = abs(prev_errors['centrality'] - status.centrality_error) < 0.02 * gap_tol

    if primal_stalled and dual_stalled and centrality_stalled:
        if status.verbose:
            print("============================================\n Progress stalled! Entering finishing phase.\n============================================")
        return True
    return False


def _ipm_check_convergence(status, finishing_steps, ZX, TX, abs_tol, max_refinement):
    """Checks for final convergence and updates the finishing step counter."""
    if not status.is_last_iter:
        return status, finishing_steps

    if (abs(ZX) + abs(TX) < abs_tol and
            status.primal_error < abs_tol and
            status.dual_error < abs_tol):
        if status.verbose:
            print("info   | finish   | accepted=absolute | action=stop")
        finishing_steps = 0
    else:
        finishing_steps -= 1
        status.boundary_val = 0.001 * (1 - (finishing_steps / max_refinement))
        if finishing_steps == 1:
            status.kkt_iterations += 1

    return status, finishing_steps


def _ipm_finish_merit(status, ZX, TX):
    raw_slack = abs(ZX) + abs(TX)
    primal_sq = status.primal_feas_norm ** 2
    dual_sq = status.dual_feas_norm ** 2
    return max(raw_slack, primal_sq, dual_sq)


def _tt_ineq_complementarity(X_tt, T_tt, status):
    if status.ineq_status is not IneqStatus.ACTIVE or T_tt is None:
        return 0.0
    return abs(tt_inner_prod(X_tt, T_tt) + status.ineq_boundary_val*tt_entrywise_sum(T_tt))


def _tt_polish_dual_y(obj_tt, Z_tt, T_tt, status):
    target = tt_rank_reduce(tt_add(tt_reshape(Z_tt, (4,)), obj_tt), status.rounding.residual_dual(status))
    if status.ineq_status is IneqStatus.ACTIVE and T_tt is not None:
        target = tt_rank_reduce(tt_add(target, tt_reshape(T_tt, (4,))), status.rounding.residual_dual(status))
    projected = tt_sub(target, tt_fast_matrix_vec_mul(status.lag_map_y, target, status.eps))
    projected = tt_rank_reduce(projected, status.rounding.residual_dual(status))
    return tt_reshape(_tt_symmetrise(tt_reshape(projected, (2, 2)), status.rounding.update_z_round_tol(status)), (4,))


def _tt_postprocess_ineq_solution(obj_tt, lin_op_tt_adj, X_tt, Y_tt, T_tt, Z_tt, ineq_mask, status):
    if status.ineq_status is not IneqStatus.ACTIVE or T_tt is None:
        return X_tt, Y_tt, T_tt, Z_tt, status

    t0 = time.time()
    dual_before = tt_norm(tt_compute_dual_feasibility(obj_tt, lin_op_tt_adj, Z_tt, Y_tt, T_tt, status))
    slack_before = _tt_ineq_complementarity(X_tt, T_tt, status)
    rank_t_before = _tt_rank_sum(T_tt)

    T_candidate = tt_rank_reduce(
        tt_fast_hadamard(ineq_mask, T_tt, status.eps),
        status.rounding.update_t_round_tol(status)
    )
    T_candidate = _tt_budgeted_mask_symmetrise(T_candidate, ineq_mask, status.rounding.update_t_round_tol(status))
    if _tt_rank_sum(T_candidate) > rank_t_before:
        T_candidate = T_tt

    Y_candidate = _tt_polish_dual_y(obj_tt, Z_tt, T_candidate, status)
    dual_after = tt_norm(tt_compute_dual_feasibility(obj_tt, lin_op_tt_adj, Z_tt, Y_candidate, T_candidate, status))
    slack_after = _tt_ineq_complementarity(X_tt, T_candidate, status)

    before_merit = max(dual_before**2, slack_before)
    after_merit = max(dual_after**2, slack_after)
    if after_merit <= before_merit * (1 + 1e-6) + 1e-12:
        status.dual_feas_norm = dual_after
        status.dual_error = dual_after / status.dual_error_normalisation
        status.is_dual_feasible = status.dual_error < status.feasibility_tol
        status.ineq_slack = slack_after
        status.mu = (status.psd_slack + status.ineq_slack) / (2 ** status.dim + status.num_ineq_constraints)
        _ipm_trace(
            status,
            "post",
            f"accepted T/Y polish dual={dual_before:.2e}->{dual_after:.2e} "
            f"tx={slack_before:.2e}->{slack_after:.2e} rT={rank_t_before}->{_tt_rank_sum(T_candidate)}",
            t0
        )
        return X_tt, Y_candidate, T_candidate, Z_tt, status

    _ipm_trace(
        status,
        "post",
        f"rejected T/Y polish dual={dual_before:.2e}->{dual_after:.2e} "
        f"tx={slack_before:.2e}->{slack_after:.2e}",
        t0
    )
    return X_tt, Y_tt, T_tt, Z_tt, status


def _tt_copy_state(X_tt, Y_tt, T_tt, Z_tt, status):
    def copy_train(tt):
        return None if tt is None else [core.copy() for core in tt]
    return copy_train(X_tt), copy_train(Y_tt), copy_train(T_tt), copy_train(Z_tt), copy.copy(status)


def _tt_reset_step_eigen_warm_starts(status):
    status.eigen_x0 = None
    status.eigen_z0 = None
    status.eigen_xt0 = None
    status.eigen_zt0 = None


def _ipm_trace(status, stage, message="", t0=None):
    if not getattr(status, "trace_verbose", False):
        return
    details = message
    if t0 is not None:
        elapsed = f"dt={time.time() - t0:.2f}s"
        details = f"{details} | {elapsed}" if details else elapsed
    print(f"trace  | {stage:<12} | {details}", flush=True)


def _tt_rank_peak(tt):
    if tt is None:
        return "-"
    return str(max(tt_ranks(tt)))


def _ipm_log_iteration(iteration, status, X_tt, Y_tt, Z_tt, T_tt):
    """Prints a compact progress line for the current iteration."""
    phase = "repair" if status.primal_restoration else ("finish" if status.is_last_iter else "main")
    direction = "AHO" if status.aho_direction else "XZ"
    feas_flags = "".join([
        "C" if status.is_central else "c",
        "P" if status.is_primal_feasible else "p",
        "D" if status.is_dual_feasible else "d",
    ])
    ineq_status = status.ineq_status.name.lower()
    ranks = "/".join([
        _tt_rank_peak(X_tt),
        _tt_rank_peak(Y_tt),
        _tt_rank_peak(Z_tt),
        _tt_rank_peak(T_tt),
    ])
    print(
        f"it {iteration - 1:03d} | {phase:<6} | {direction:<3} | ineq={ineq_status:<8} | "
        f"mu={status.mu:.2e} eta={status.eta:.2e} sigma={status.sigma:.2e} | "
        f"err c/p/d={status.centrality_error:.2e}/{status.primal_error:.2e}/{status.dual_error:.2e} | "
        f"ok={feas_flags} | rmax X/Y/Z/T={ranks}",
        flush=True,
    )


def tt_ipm(
    lag_maps,
    obj_tt,
    lin_op_tt,
    bias_tt,
    ineq_mask=None,
    max_iter=100,
    max_refinement=5,
    warm_up=3,
    gap_tol=1e-4,
    aho_direction=True,
    op_tol=1e-5,
    abs_tol=8e-4,
    eps=1e-12,
    mals_restarts=3,
    r_max=1000,
    lambdaStar=1,
    lambdaStarIneq=1,
    # Backward-compat deprecated aliases:
    epsilonDash=None,
    epsilonDashineq=None,
    verbose=False,
    solver_verbose=False,
    trace_verbose=False,
    rounding_update_budget_growth=1.25,
    delta_mul_kkt_weight=0.25,
    combine_ty=False,
    eq_mask=None
):
    dim = len(obj_tt)
    centrality_tol = gap_tol / np.sqrt(dim) # for larger problems we need to be closer
    feasibility_tol = 2*gap_tol
    status = IPMStatus(
        len(obj_tt),
        feasibility_tol,
        centrality_tol,
        op_tol,
        eps,
        aho_direction,
        False,
        np.inf,
        False,
        np.inf,
        False,
        np.inf,
        np.inf,
        False,
        IneqStatus.NOT_IN_USE if ineq_mask is None else IneqStatus.ACTIVE,
        solver_verbose,
        1,
        1,
        r_max
    )
    status.trace_verbose = trace_verbose
    status.rounding = RoundingController(update_budget_growth=rounding_update_budget_growth)
    delta_mul_kkt_weight = float(delta_mul_kkt_weight)
    if not np.isfinite(delta_mul_kkt_weight) or delta_mul_kkt_weight <= 0:
        raise ValueError("Lagrange multiplier KKT weight must be finite and positive")
    multiplier_block_weights = np.array([delta_mul_kkt_weight, 1.0, 1.0], dtype=np.float64)
    ineq_block_weights = np.array([delta_mul_kkt_weight, 1.0, 1.0, delta_mul_kkt_weight], dtype=np.float64)
    lag_maps = {key: tt_rank_reduce(value, eps=eps) for key, value in lag_maps.items()}
    obj_tt = tt_rank_reduce(obj_tt, eps=eps)
    lin_op_tt = tt_rank_reduce(lin_op_tt, eps=eps)
    bias_tt = tt_rank_reduce(bias_tt, eps=eps)
    if ineq_mask is not None:
        ineq_mask = tt_rank_reduce(ineq_mask, eps=eps)
    if eq_mask is not None:
        eq_mask = tt_rank_reduce(eq_mask, eps=eps)

    combine_ty = bool(combine_ty and ineq_mask is not None and eq_mask is not None)
    if combine_ty and abs(tt_inner_prod(eq_mask, ineq_mask)) > max(1e-8, 100 * eps):
        if verbose:
            print("warn   | ipm      | combine_ty=false | reason=mask-overlap", flush=True)
        combine_ty = False

    status.primal_error_normalisation = 1 + tt_norm(bias_tt)
    status.dual_error_normalisation = 1 + tt_norm(obj_tt)
    status.combine_ty = combine_ty
    if combine_ty:
        status.eq_mask = eq_mask
        status.ty_idle_mask = tt_rank_reduce(tt_sub(tt_sub(tt_one_matrix(dim), eq_mask), ineq_mask), eps=eps)
        status.lag_map_ty = tt_diag_op(status.ty_idle_mask, eps)

    lhs_skeleton = TTBlockMatrix()
    lhs_skeleton[1, 2] = tt_reshape(tt_identity(2 * dim), (4, 4))
    solver_ineq = lambda lhs, rhs, x0, nwsp, restriction, termination_tol, strict_first_attempt=False: tt_restarted_block_amen(
        lhs,
        rhs,
        rank_restriction=restriction,
        x0=x0,
        local_solver=_ipm_local_solver_ineq,
        op_tol=op_tol,
        termination_tol=termination_tol,
        num_restarts=mals_restarts,
        inner_m=nwsp,
        verbose=solver_verbose,
        strict_first_attempt=strict_first_attempt,
        block_weights=ineq_block_weights
    )
    solver_ty = lambda lhs, rhs, x0, nwsp, restriction, termination_tol, strict_first_attempt=False: tt_restarted_block_amen(
        lhs,
        rhs,
        rank_restriction=restriction,
        x0=x0,
        local_solver=_ipm_local_solver_ty,
        op_tol=op_tol,
        termination_tol=termination_tol,
        num_restarts=mals_restarts,
        inner_m=nwsp,
        verbose=solver_verbose,
        strict_first_attempt=strict_first_attempt,
        block_weights=multiplier_block_weights
    )
    solver_eq = lambda lhs, rhs, x0, nwsp, restriction, termination_tol, strict_first_attempt=False: tt_restarted_block_amen(
        lhs,
        rhs,
        rank_restriction=restriction,
        x0=x0,
        local_solver=_ipm_local_solver,
        op_tol=op_tol,
        termination_tol=termination_tol,
        num_restarts=mals_restarts,
        inner_m=nwsp,
        verbose=solver_verbose,
        strict_first_attempt=strict_first_attempt,
        block_weights=multiplier_block_weights
    )
    if status.ineq_status is IneqStatus.ACTIVE:
        solver = solver_ty if combine_ty else solver_ineq
        status.num_ineq_constraints = tt_inner_prod(ineq_mask, ineq_mask)
        status.compl_ineq_mask = tt_rank_reduce(tt_sub(tt_one_matrix(dim), ineq_mask), eps=eps)
        status.lag_map_t = lag_maps["t"]
        if not combine_ty:
            lhs_skeleton.add_alias((1, 2), (1, 3))
    else:
        solver = solver_eq
        status.num_ineq_constraints = 0

    # KKT-system prep
    lin_op_tt_adj = tt_transpose(lin_op_tt)
    if combine_ty:
        lhs_skeleton[0, 0] = status.lag_map_ty
        lhs_skeleton[0, 1] = tt_scale(-1, lin_op_tt)
        lhs_skeleton[1, 0] = tt_scale(-1, lin_op_tt_adj)
    else:
        lhs_skeleton[0, 1] = tt_scale(-1, lin_op_tt)
        lhs_skeleton.add_alias((0, 1), (1, 0), is_transpose=True)
        lhs_skeleton[0, 0] = lag_maps["y"]
    status.lag_map_y = lag_maps["y"]

    X_tt, Y_tt, Z_tt, T_tt = _initialise(ineq_mask, status, dim, lambdaStar, lambdaStarIneq)

    iteration = 0
    finishing_steps = max_refinement
    prev_errors = {'primal': np.inf, 'dual': np.inf, 'centrality': np.inf}
    finish_prev_state = None
    finish_prev_merit = np.inf
    lhs = lhs_skeleton

    while finishing_steps > 0:
        iteration += 1
        iter_t0 = time.time()
        status.aho_direction = (iteration > warm_up)
        _ipm_trace(status, "iter", f"start it={iteration - 1:03d} dir={'AHO' if status.aho_direction else 'XZ'} ineq={status.ineq_status.name.lower()}")
        if max_iter - max_refinement == iteration - 1 and not status.is_last_iter:
            print("warn   | ipm      | limit=max_iter | action=finish", flush=True)
            status.is_last_iter = True
        ZX = tt_inner_prod(Z_tt, X_tt)
        if status.ineq_status is IneqStatus.ACTIVE:
            raw_TX = tt_inner_prod(X_tt, T_tt)
            status.ineq_boundary_val = _tt_update_ineq_boundary(status, ZX, raw_TX, T_tt)
            TX = raw_TX + status.ineq_boundary_val*tt_entrywise_sum(T_tt)
        else:
            TX = 0
        status.iteration = iteration
        status.psd_slack = abs(ZX)
        status.ineq_slack = abs(TX)
        status.mu = np.divide(abs(ZX) + abs(TX), (2 ** dim + (status.ineq_status is IneqStatus.ACTIVE)*status.num_ineq_constraints))
        status.centrl_error_normalisation = 1 + abs(tt_inner_prod(obj_tt, tt_reshape(X_tt, (4, ))))
        status.centrality_error = status.mu / status.centrl_error_normalisation
        status.is_central = np.less(status.centrality_error, centrality_tol)
        status.eta = max(min(status.eta, 2*status.mu), status.op_tol)

        t0 = time.time()
        lhs_matrix_tt, rhs_vec_tt, status = tt_infeasible_newton_system(
            lhs,
            obj_tt,
            X_tt,
            Y_tt,
            Z_tt,
            T_tt,
            lin_op_tt,
            lin_op_tt_adj,
            bias_tt,
            ineq_mask,
            status
        )
        _ipm_trace(status, "system", f"rhs_rows={list(rhs_vec_tt.keys())}", t0)

        if status.is_last_iter:
            finish_merit = _ipm_finish_merit(status, ZX, TX)
            if finish_prev_state is not None and finish_merit > finish_prev_merit * (1 + 1e-8) + 1e-12:
                X_tt, Y_tt, T_tt, Z_tt, status = finish_prev_state
                iteration -= 1
                if verbose:
                    print(f"warn   | finish   | merit=worse | prev={finish_prev_merit:.2e} current={finish_merit:.2e} | action=restore-stop", flush=True)
                break
            finish_prev_merit = finish_merit
            finish_prev_state = _tt_copy_state(X_tt, Y_tt, T_tt, Z_tt, status)
        else:
            finish_prev_state = None
            finish_prev_merit = np.inf

        if verbose:
            _ipm_log_iteration(iteration, status, X_tt, Y_tt, Z_tt, T_tt)

        status, finishing_steps = _ipm_check_convergence(
            status, finishing_steps, ZX, TX, abs_tol, max_refinement
        )
        if finishing_steps == 0:
            iteration -= 1
            break

        t0 = time.time()
        x_step_size, z_step_size, Delta_X_tt, Delta_Y_tt, Delta_Z_tt, Delta_T_tt, status = _tt_ipm_newton_step(
            lhs_matrix_tt,
            rhs_vec_tt,
            ineq_mask,
            X_tt,
            Z_tt,
            T_tt,
            ZX,
            TX,
            status,
            solver
        )
        _ipm_trace(status, "newton-all", "returned", t0)

        if (Delta_X_tt is None and Delta_Z_tt is None) or (x_step_size < 1e-5 and z_step_size < 1e-5):
            if status.is_last_iter:
                break
            else:
                print("warn   | ipm      | event=psd-boundary | action=finish", flush=True)
                status.is_last_iter = True
        else:
            t0 = time.time()
            X_tt = _tt_budgeted_psd_symmetrise(tt_add(X_tt, tt_scale(x_step_size, Delta_X_tt)), status.rounding.update_x_round_tol(status))
            Z_tt = _tt_budgeted_psd_symmetrise(tt_add(Z_tt, tt_scale(z_step_size, Delta_Z_tt)), status.rounding.update_z_round_tol(status))

            Y_tt = tt_rank_reduce(tt_add(Y_tt, tt_scale(z_step_size, Delta_Y_tt)), status.eps)
            Y_tt = tt_reshape(_tt_symmetrise(tt_reshape(tt_sub(Y_tt, tt_fast_matrix_vec_mul(status.lag_map_y, Y_tt, status.eps)), (2, 2)), status.rounding.update_z_round_tol(status)), (4, ))

            if status.ineq_status is IneqStatus.ACTIVE and Delta_T_tt is not None:
                T_tt = _tt_budgeted_mask_symmetrise(tt_add(T_tt, tt_scale(z_step_size, Delta_T_tt)), ineq_mask, status.rounding.update_t_round_tol(status))
            _ipm_trace(status, "update", f"rmax X/Y/Z/T={_tt_rank_peak(X_tt)}/{_tt_rank_peak(Y_tt)}/{_tt_rank_peak(Z_tt)}/{_tt_rank_peak(T_tt)}", t0)
            if status.ineq_next_status is IneqStatus.INACTIVE and status.ineq_status is IneqStatus.ACTIVE:
                _ipm_trace(status, "ineq", "switch active->inactive")
                solver = solver_eq
                lhs = lhs_skeleton.get_submatrix(2, 2)
                status.mals_delta0 = None
                _tt_reset_step_eigen_warm_starts(status)
                status.ineq_status = IneqStatus.INACTIVE
                status.ineq_next_status = None
                status.ineq_full_step_streak = 0
                status.ineq_last_switch_iter = iteration
            elif status.ineq_next_status is IneqStatus.ACTIVE and status.ineq_status is IneqStatus.INACTIVE:
                _ipm_trace(status, "ineq", "switch inactive->active")
                solver = solver_ineq
                lhs = lhs_skeleton
                status.mals_delta0 = None
                _tt_reset_step_eigen_warm_starts(status)
                status.ineq_status = IneqStatus.ACTIVE
                status.ineq_next_status = None
                status.ineq_full_step_streak = 0
                status.ineq_last_switch_iter = iteration

        if _ipm_check_for_stalled_progress(prev_errors, status, gap_tol):
            status.is_last_iter = True

        _ipm_trace(status, "iter", f"done it={iteration - 1:03d}", iter_t0)

        prev_errors['primal'] = status.primal_error
        prev_errors['dual'] = status.dual_error
        prev_errors['centrality'] = status.centrality_error

    X_tt, Y_tt, T_tt, Z_tt, status = _tt_postprocess_ineq_solution(
        obj_tt, lin_op_tt_adj, X_tt, Y_tt, T_tt, Z_tt, ineq_mask, status
    )
    return _ipm_format_output(X_tt, Y_tt, T_tt, Z_tt, iteration, status)
