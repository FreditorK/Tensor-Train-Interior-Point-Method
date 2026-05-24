import sys
import os
import numpy as np
import traceback
import scipy.linalg as la

sys.path.append(os.getcwd() + '/../')

from src.tt_ops import *
from src.tt_als import cached_einsum, TTBlockMatrix, TTBlockVector, tt_mat_vec_mul, tt_max_generalised_eigen, tt_min_eig, tt_mat_mat_mul,tt_restarted_block_amen
from dataclasses import dataclass
from enum import Enum
from petsc4py import PETSc

import warnings
warnings.simplefilter("error")


_DENSE_FALLBACK_EXCEPTIONS = (la.LinAlgWarning, la.LinAlgError, np.linalg.LinAlgError)


def _print_unexpected_dense_failure(e):
    if isinstance(e, _DENSE_FALLBACK_EXCEPTIONS):
        return
    print(e)
    last = traceback.extract_tb(e.__traceback__)[-1]
    print(f"\t{type(e).__name__} in {last.filename},\n\tline {last.lineno}: {last.line.strip()}")


def _expected_newton_fallback(e):
    return isinstance(e, RuntimeError) and (
        "Number of restarts exhausted" in str(e)
        or "Absolute tolerance already reached" in str(e)
    )

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
                cached_einsum("lsr,smnS,LSR->lmLrnR",
                              XAX_k_21[r_i:r_ip1, :, r_i:r_ip1],
                              block_A_k_21,
                              XAX_k1_21
                ).reshape((r_ip1-r_i)*self.base_size, (r_ip1-r_i)*self.base_size)
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
                cached_einsum("lsr,smnS,LSR->lmLrnR",
                              XAX_k_00[r_i:r_ip1, :, r_i:r_ip1],
                              block_A_k_00,
                              XAX_k1_00
                ).reshape((r_ip1-r_i)*self.base_size, (r_ip1-r_i)*self.base_size)
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
        x_np = x.getArray()
        x1, x2 = x_np[:self.m], x_np[self.m:]
        y2 = self.LZInv.solve(x2)
        y1 = self.KyInv.solve(x1)
        y_np = np.concatenate([y1, y2])
        y.setArray(y_np)

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
        np.copyto(self.x_buffer, x.array_r)
        y_np = self.matvec_object.matvec(self.x_buffer)
        y.array_w[:] = y_np

    def solve_system(self, matvec_object, rhs_np, shape):
        rhs_np = np.ascontiguousarray(rhs_np, dtype=np.float64)
        self.matvec_object = matvec_object
        self.shape = shape
        self.x_buffer = np.empty(shape[1], dtype=rhs_np.dtype)

        self.A_shell = PETSc.Mat().createPython(self.shape, comm=PETSc.COMM_WORLD)
        self.A_shell.setPythonContext(self)
        self.A_shell.setUp()
        self.ksp.setOperators(self.A_shell)

        b_petsc = PETSc.Vec().createWithArray(rhs_np, comm=PETSc.COMM_WORLD)
        x_petsc = PETSc.Vec().createWithArray(np.zeros_like(rhs_np), comm=PETSc.COMM_WORLD)
        try:
            self.ksp.solve(b_petsc, x_petsc)
            sol = x_petsc.getArray().copy()
        finally:
            b_petsc.destroy()
            x_petsc.destroy()
        return sol

    def destroy(self, _=None):
        if hasattr(self, "ksp") and self.ksp:
            self.ksp.setOperators(None)
            self.ksp.reset()
            self.ksp.destroy()
            self.ksp = None
        if hasattr(self, "A_shell") and self.A_shell:
            self.A_shell = None
        self.matvec_object = None
        self.shape = None
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

def forward_backward_sub(L, b, overwrite_b=False):
    y = scp.linalg.solve_triangular(L, b, lower=True, check_finite=False, overwrite_b=overwrite_b)
    x = scp.linalg.solve_triangular(L.T, y, lower=False, check_finite=False, overwrite_b=True)
    return x

def _ipm_local_solver(XAX_k, block_A_k, XAX_k1, Xb_k, block_b_k, Xb_k1, previous_solution, size_limit, dense_solve=True, rtol=1e-5):
    x_shape = previous_solution.shape
    m = x_shape[0] * x_shape[2] * x_shape[3]
    rhs = np.empty_like(previous_solution)
    rhs[:, 0] = cached_einsum('br,bmB,BR->rmR', Xb_k[0], block_b_k[0], Xb_k1[0]) if 0 in block_b_k else 0
    rhs[:, 1] = cached_einsum('br,bmB,BR->rmR', Xb_k[1], block_b_k[1], Xb_k1[1]) if 1 in block_b_k else 0
    rhs[:, 2] = cached_einsum('br,bmB,BR->rmR', Xb_k[2], block_b_k[2], Xb_k1[2]) if 2 in block_b_k else 0
    norm_rhs = max(np.linalg.norm(rhs), 1e-10)
    block_res_old = np.linalg.norm(block_A_k.block_local_product(XAX_k, XAX_k1, previous_solution).__isub__(rhs)) / norm_rhs
    direct_solve_failure = not dense_solve
    dense_solve = (np.sqrt(x_shape[0]*x_shape[3]) <= size_limit) and dense_solve

    if block_res_old < rtol:
        return previous_solution, block_res_old, block_res_old, rhs, norm_rhs, direct_solve_failure

    inv_I = np.divide(1, cached_einsum('lsr,smnS,LSR->lmL', XAX_k[1, 2], block_A_k[1, 2], XAX_k1[1, 2]))
    
    if dense_solve:
        try:
            mR_p = rhs[:, 0].reshape(m, 1)
            mR_d = rhs[:, 1].reshape(m, 1)
            mR_c = rhs[:, 2].reshape(m, 1)
            L_X_I_inv = cached_einsum('lsr,smnS,LSR->lmLrnR', XAX_k[2, 2], block_A_k[2, 2], XAX_k1[2, 2]).reshape(m, m)
            L_X_I_inv *= inv_I.reshape(1, -1)
            mL_eq = cached_einsum('lsr,smnS,LSR->lmLrnR', XAX_k[0, 1], block_A_k[0, 1], XAX_k1[0, 1]).reshape(m, m)
            L_L_Z = scp.linalg.cholesky(
                cached_einsum('lsr,smnS,LSR->lmLrnR', XAX_k[2, 1], block_A_k[2, 1], XAX_k1[2, 1]).reshape(m, m),
                check_finite=False, lower=True, overwrite_a=True
            )
            b = mR_p - mL_eq @ forward_backward_sub(L_L_Z, mR_c - L_X_I_inv @ mR_d, overwrite_b=True)
            A = forward_backward_sub(L_L_Z, L_X_I_inv, overwrite_b=True)
            np.matmul(A, mL_eq.T, out=A)
            np.matmul(mL_eq, A, out=A)
            A += cached_einsum('lsr,smnS,LSR->lmLrnR',XAX_k[0, 0], block_A_k[0, 0], XAX_k1[0, 0]).reshape(m, m)
            A.flat[::A.shape[1] + 1] += 1e-11
            solution_now = np.empty(x_shape)
            solution_now[:, 0] = scp.linalg.solve(A, b, check_finite=False, overwrite_a=True, overwrite_b=True, assume_a="gen").reshape(x_shape[0], x_shape[2], x_shape[3])
            solution_now[:, 2] = (
                mR_d - cached_einsum('lsr,smnS,LSR,lmL->rnR', XAX_k[0, 1], block_A_k[0, 1], XAX_k1[0, 1], solution_now[:, 0]).reshape(-1, 1)
                ).__imul__(inv_I.reshape(-1, 1)).reshape(x_shape[0], x_shape[2], x_shape[3])
            solution_now[:, 1] = forward_backward_sub(
                L_L_Z, 
                mR_c - cached_einsum('lsr,smnS,LSR,rnR->lmL', XAX_k[2, 2], block_A_k[2, 2], XAX_k1[2, 2], solution_now[:, 2]).reshape(-1, 1), 
                overwrite_b=True
                ).reshape(x_shape[0], x_shape[2], x_shape[3])
        except Exception as e:
            _print_unexpected_dense_failure(e)
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
        local_rhs[1] -= cached_einsum('lsr,smnS,LSR,rnR->lmL', XAX_k[2, 2], block_A_k[2, 2], XAX_k1[2, 2], inv_I*rhs[:, 1])
        local_rhs_norm = np.linalg.norm(local_rhs)
        local_vec = matvec_wrapper.matvec(np.transpose(previous_solution[:, :2], (1, 0, 2, 3)).flatten()).reshape(2, x_shape[0], x_shape[2], x_shape[3])
        local_rhs_norm_prime = np.linalg.norm(local_rhs - local_vec)
        use_prev_sol = (local_rhs_norm_prime < local_rhs_norm)
        if use_prev_sol:
            local_rhs -= local_vec
        
        num_iters = min(m, 100)
        outer_k = max(num_iters // 10, 3)
        large_scale_solver = LGMRESSolver(rtol=rtol, restart=num_iters, outer_k=outer_k)
        try:
            solution_now = large_scale_solver.solve_system(matvec_wrapper, local_rhs.ravel(), (2*m, 2*m))
        finally:
            large_scale_solver.destroy()
        solution_now = np.transpose(solution_now.reshape(2, x_shape[0], x_shape[2], x_shape[3]), (1, 0, 2, 3))

        if use_prev_sol:
            solution_now[:, :2] += previous_solution[:, :2]
            
        z = inv_I * (rhs[:, 1] - cached_einsum('lsr,smnS,LSR,lmL->rnR', XAX_k[0, 1], block_A_k[0, 1], XAX_k1[0, 1], solution_now[:, 0]))
        solution_now = np.concatenate((solution_now, z.reshape(x_shape[0], 1, x_shape[2], x_shape[3])), axis=1)

    block_res_new = np.linalg.norm(block_A_k.block_local_product(XAX_k, XAX_k1, solution_now).__isub__(rhs)) / norm_rhs

    if block_res_old < block_res_new:
        solution_now = previous_solution        

    return solution_now, block_res_old, min(block_res_old, block_res_new), rhs, norm_rhs, direct_solve_failure

def _ipm_local_solver_ineq(XAX_k, block_A_k, XAX_k1, Xb_k, block_b_k, Xb_k1, previous_solution, size_limit, dense_solve=True, rtol=1e-5):
    x_shape = previous_solution.shape
    m = x_shape[0] * x_shape[2] * x_shape[3]
    rhs = np.empty_like(previous_solution)
    rhs[:, 0] = cached_einsum('br,bmB,BR->rmR', Xb_k[0], block_b_k[0], Xb_k1[0]) if 0 in block_b_k else 0
    rhs[:, 1] = cached_einsum('br,bmB,BR->rmR', Xb_k[1], block_b_k[1], Xb_k1[1]) if 1 in block_b_k else 0
    rhs[:, 2] = cached_einsum('br,bmB,BR->rmR', Xb_k[2], block_b_k[2], Xb_k1[2]) if 2 in block_b_k else 0
    rhs[:, 3] = cached_einsum('br,bmB,BR->rmR', Xb_k[3], block_b_k[3], Xb_k1[3]) if 3 in block_b_k else 0
    norm_rhs = max(np.linalg.norm(rhs), 1e-10)
    block_res_old = np.linalg.norm(block_A_k.block_local_product(XAX_k, XAX_k1, previous_solution).__isub__(rhs)) / norm_rhs
    direct_solve_failure = not dense_solve
    dense_solve = (np.sqrt(x_shape[0]*x_shape[3]) <= 0.95*size_limit) and dense_solve

    if block_res_old < rtol:
        return previous_solution, block_res_old, block_res_old, rhs, norm_rhs, direct_solve_failure

    inv_I = np.divide(1, cached_einsum('lsr,smnS,LSR->lmL', XAX_k[1, 2], block_A_k[1, 2], XAX_k1[1, 2]))
            
    if dense_solve:
        try:
            L_L_Z = scp.linalg.cholesky(
                cached_einsum('lsr,smnS,LSR->lmLrnR', XAX_k[2, 1], block_A_k[2, 1], XAX_k1[2, 1]).reshape(m, m),
                check_finite=False, lower=True,  overwrite_a=True
            )
            mR_p = rhs[:, 0].reshape(m, 1)
            mR_d = rhs[:, 1].reshape(m, 1)
            mR_c = rhs[:, 2].reshape(m, 1)
            mR_t = rhs[:, 3].reshape(m, 1)
            L_L_Z_inv_mR_c = forward_backward_sub(L_L_Z, rhs[:, 2].reshape(m, 1))
            L_L_Z_inv_L_X = forward_backward_sub(L_L_Z, cached_einsum('lsr,smnS,LSR->lmLrnR', XAX_k[2, 2], block_A_k[2, 2], XAX_k1[2, 2]).reshape(m, m), overwrite_b=True)
            mL_eq = cached_einsum('lsr,smnS,LSR->lmLrnR', XAX_k[0, 1], block_A_k[0, 1], XAX_k1[0, 1]).reshape(m, m)
            T_op = cached_einsum('lsr,smnS,LSR->lmLrnR', XAX_k[3, 1], block_A_k[3, 1], XAX_k1[3, 1]).reshape(m, m)
            u = mR_p - mL_eq @ (L_L_Z_inv_mR_c - (L_L_Z_inv_L_X * inv_I.reshape(1, -1)) @ mR_d)
            v = mR_t - T_op @ (L_L_Z_inv_mR_c - (L_L_Z_inv_L_X * inv_I.reshape(1, -1)) @ mR_d)
            A = cached_einsum('lsr,smnS,LSR->lmLrnR', XAX_k[0, 0], block_A_k[0, 0],XAX_k1[0, 0]).reshape(m, m).__iadd__(mL_eq @ (L_L_Z_inv_L_X * inv_I.reshape(1, -1)) @ mL_eq.T)
            D = cached_einsum('lsr,smnS,LSR->lmLrnR', XAX_k[3, 3], block_A_k[3, 3], XAX_k1[3, 3]).reshape(m, m).__iadd__(T_op @ L_L_Z_inv_L_X)
            D.flat[::D.shape[1] + 1] += 1e-11
            np.matmul(T_op, L_L_Z_inv_L_X * inv_I.reshape(1, -1), out=T_op)
            np.matmul(T_op, mL_eq.T, out=T_op)
            np.matmul(mL_eq, L_L_Z_inv_L_X, out=mL_eq)
            Dlu, Dpiv = scp.linalg.lu_factor(D, check_finite=False, overwrite_a=True)
            rhs_l = u.__isub__(mL_eq @ scp.linalg.lu_solve((Dlu, Dpiv), v, check_finite=False))
            lhs_l = A.__isub__(mL_eq.__imatmul__(scp.linalg.lu_solve((Dlu, Dpiv), T_op, check_finite=False)))
            y = scp.linalg.lu_solve(scp.linalg.lu_factor(lhs_l, check_finite=False, overwrite_a=True), rhs_l, check_finite=False, overwrite_b=True)
            solution_now = np.empty(x_shape)
            solution_now[:, 0] = y.reshape(x_shape[0], x_shape[2], x_shape[3])
            solution_now[:, 3] = scp.linalg.lu_solve((Dlu, Dpiv), v.__isub__(T_op @ y), check_finite=False, overwrite_b=True).reshape(x_shape[0], x_shape[2], x_shape[3])
            solution_now[:, 2] = (
                mR_d - cached_einsum('lsr,smnS,LSR,lmL->rnR', XAX_k[0, 1], block_A_k[0, 1], XAX_k1[0, 1], solution_now[:, 0]).reshape(-1, 1)
                ).__imul__(inv_I.reshape(-1, 1)).reshape(x_shape[0], x_shape[2], x_shape[3]).__isub__(solution_now[:, 3])
            solution_now[:, 1] = forward_backward_sub(
                L_L_Z, 
                mR_c - cached_einsum('lsr,smnS,LSR,rnR->lmL', XAX_k[2, 2], block_A_k[2, 2], XAX_k1[2, 2], solution_now[:, 2]).reshape(-1, 1), 
                overwrite_b=True
                ).reshape(x_shape[0], x_shape[2], x_shape[3])

        except Exception as e:
            _print_unexpected_dense_failure(e)
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
        local_rhs[1] = rhs[:, 2] - cached_einsum('lsr,smnS,LSR,rnR->lmL', XAX_k[2, 2], block_A_k[2, 2],
                                                  XAX_k1[2, 2], inv_I * rhs[:, 1])
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
        try:
            solution_now = large_scale_solver.solve_system(matvec_wrapper, local_rhs.ravel(), (3*m, 3*m))
        finally:
            large_scale_solver.destroy()
        solution_now = np.transpose(solution_now.reshape(3, x_shape[0], x_shape[2], x_shape[3]),
                                    (1, 0, 2, 3)) 
        
        if use_prev_sol:
            solution_now[:, 0] += previous_solution[:, 0]
            solution_now[:, 1] += previous_solution[:, 1]
            solution_now[:, 2] += previous_solution[:, 3]

        z = inv_I * (
                    rhs[:, 1] - cached_einsum('lsr,smnS,LSR,lmL->rnR', XAX_k[0, 1], block_A_k[0, 1], XAX_k1[0, 1],
                                              solution_now[:, 0])) - solution_now[:, 2]
        solution_now = np.concatenate(
            (solution_now[:, :2], z.reshape(x_shape[0], 1, x_shape[2], x_shape[3]), solution_now[:, None, 2]), axis=1)
        
    block_res_new = np.linalg.norm(block_A_k.block_local_product(XAX_k, XAX_k1, solution_now) - rhs) / norm_rhs

    if block_res_old < block_res_new:
        solution_now = previous_solution

    return solution_now, block_res_old, min(block_res_old, block_res_new), rhs, norm_rhs, direct_solve_failure


def tt_compute_primal_feasibility(lin_op_tt, bias_tt, X_tt, status):
    tol = status.rounding.primal_residual(status)
    primal_feas = tt_rank_reduce(
        tt_sub(tt_mat_vec_mul(lin_op_tt, tt_reshape(X_tt, (4,)), tol, status.eps), bias_tt),
        tol
    )
    return primal_feas


def tt_compute_dual_feasibility(obj_tt, lin_op_tt_adj, Z_tt, Y_tt, T_tt, status):
    tol = status.rounding.dual_residual(status)
    dual_feas = tt_rank_reduce(
        tt_sub(
            tt_fast_matrix_vec_mul(lin_op_tt_adj, Y_tt, tol),
            tt_rank_reduce(tt_add(tt_reshape(Z_tt, (4,)), obj_tt), tol)
        ),
        tol
    )
    if status.ineq_status is IneqStatus.ACTIVE and T_tt is not None:
        dual_feas = tt_rank_reduce(tt_sub(dual_feas, tt_reshape(T_tt, (4,))), tol)

    return dual_feas


def tt_compute_centrality(X_tt, Z_tt, status):
    tol = status.rounding.centrality_residual(status)
    if status.aho_direction:
        centrality_feas = tt_reshape(
            tt_scale(-1, _tt_symmetrise(tt_mat_mat_mul(X_tt, Z_tt, tol, status.eps), tol)),
            (4,)
        )
    else:
        centrality_feas = tt_reshape(tt_scale(-1, tt_mat_mat_mul(Z_tt, X_tt, tol, status.eps)), (4,))
    return centrality_feas


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
    status.primal_error = np.divide(tt_norm(primal_feas), status.primal_error_normalisation)
    status.is_primal_feasible = np.less(status.primal_error, status.feasibility_tol)

    dual_feas = tt_compute_dual_feasibility(obj_tt, lin_op_tt_adj, Z_tt, Y_tt, T_tt, status)
    status.dual_error = np.divide(tt_norm(dual_feas), status.dual_error_normalisation)
    status.is_dual_feasible = np.less(status.dual_error, (1 + (status.ineq_status is IneqStatus.ACTIVE))*status.feasibility_tol)

    status.is_last_iter = status.is_last_iter or (status.is_primal_feasible and status.is_dual_feasible and status.is_central)

    if status.aho_direction:
        lhs[2, 1] = tt_psd_rank_reduce(
            tt_scale(0.5, tt_add(tt_IkronM(Z_tt), tt_MkronI(Z_tt))),
            eps=status.rounding.kkt_dual(status)
        )
        lhs[2, 2] = tt_psd_rank_reduce(
            tt_scale(0.5, tt_add(tt_MkronI(X_tt), tt_IkronM(X_tt))),
            eps=status.rounding.kkt_primal(status)
        )
    else:
        lhs[2, 1] = tt_psd_rank_reduce(tt_MkronI(Z_tt), eps=status.rounding.kkt_dual(status))
        lhs[2, 2] = tt_psd_rank_reduce(tt_IkronM(X_tt), eps=status.rounding.kkt_primal(status))

    if not status.is_primal_feasible or status.is_last_iter:
        rhs[0] = primal_feas

    if not status.is_dual_feasible or status.is_last_iter:
        rhs[1] = dual_feas

    if not status.is_central or status.is_last_iter:
        rhs[2] = tt_compute_centrality(X_tt, Z_tt, status)

    if status.ineq_status is IneqStatus.ACTIVE:
        lhs[3, 1] = tt_diag_op(T_tt, status.rounding.kkt_dual(status))
        masked_X_tt = tt_rank_reduce(
            tt_add(tt_scale(status.ineq_boundary_val, ineq_mask), tt_fast_hadamard(ineq_mask, X_tt, status.eps)),
            eps=status.eps
        )
        lhs[3, 3] = tt_rank_reduce(
            tt_add(status.lag_map_t, tt_diag_op(masked_X_tt, status.eps)),
            eps=status.rounding.kkt_dual(status)
        )
        if not status.is_central or status.is_last_iter:
            rhs[3] = tt_rank_reduce(
                tt_reshape(tt_scale(-1, tt_fast_hadamard(masked_X_tt, T_tt, status.eps)), (4, )),
                eps=status.rounding.centrality_residual(status)
            )
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
        # Predictor
        if status.verbose:
            print("\n--- Predictor  step ---", flush=True)
        strict_predictor = not (status.is_last_iter or (status.is_primal_feasible and status.is_dual_feasible))
        Delta_tt, _ = solver(lhs_matrix_tt, rhs_vec_tt, status.mals_delta0, status.kkt_iterations + status.is_last_iter, status.mals_rank_restriction, status.eta, strict_predictor)
        status.mals_delta0 = Delta_tt
        Delta_X_tt = _tt_symmetrise(tt_reshape(_tt_get_block(1, Delta_tt), (2, 2)), status.eps)
        Delta_Z_tt = _tt_symmetrise(tt_reshape(_tt_get_block(2, Delta_tt), (2, 2)), status.eps)
        Delta_Y_tt = tt_rank_reduce(_tt_get_block(0, Delta_tt), eps=status.eps)
        Delta_T_tt = None
        if status.ineq_status is IneqStatus.ACTIVE:
            Delta_T_tt = tt_rank_reduce(_tt_get_block(3, Delta_tt), eps=status.eps)
            Delta_T_tt = tt_fast_hadamard(ineq_mask, tt_reshape(Delta_T_tt, (2, 2)), status.eps)

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

        if not status.is_central and not status.is_last_iter:

            DXZ = tt_inner_prod(Delta_X_tt, Delta_Z_tt)
            # Corrector
            if status.verbose:
                print(f"\n--- Centering-Corrector  step ---", flush=True)

            if status.ineq_status is IneqStatus.ACTIVE:
                mu_aff = (
                    ZX + x_step_size * z_step_size * DXZ
                    + z_step_size * tt_inner_prod(X_tt, Delta_Z_tt)
                    + x_step_size * tt_inner_prod(Delta_X_tt, Z_tt)
                    + TX + x_step_size * z_step_size * tt_inner_prod(Delta_T_tt, Delta_X_tt)
                    + z_step_size * (tt_inner_prod(X_tt, Delta_T_tt) + status.ineq_boundary_val*tt_entrywise_sum(Delta_T_tt))
                    + x_step_size * tt_inner_prod(Delta_X_tt, T_tt)
                )
                e = max(1, 3 * min(x_step_size, z_step_size) ** 2)
                status.sigma = min(0.99, max(mu_aff/(ZX + TX), 0)**e)
                if status.sigma > 1e-4:
                    rhs_vec_tt[3]  = tt_rank_reduce(tt_add(
                            tt_scale(status.sigma * status.mu, tt_reshape(ineq_mask, (4,))),
                            rhs_vec_tt.get_row(3),
                            ), status.rounding.centrality_residual(status)
                            )
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
                    status.rounding.centrality_residual(status)
                ) if status.sigma > 1e-4 else tt_rank_reduce(
                    tt_add(rhs_vec_tt.get_row(2), Delta_XZ_term),
                    status.rounding.centrality_residual(status)
                )
            else:
                rhs_vec_tt[2] = tt_rank_reduce(
                    tt_add(
                        tt_scale(status.sigma * status.mu, tt_reshape(tt_identity(len(X_tt)), (4,))),
                        rhs_vec_tt.get_row(2)
                    ),
                    status.rounding.centrality_residual(status)
                ) if status.sigma > 1e-4 else rhs_vec_tt.get_row(2)

            try:
                Delta_tt_cc, _ = solver(lhs_matrix_tt, rhs_vec_tt, status.mals_delta0, status.kkt_iterations + status.is_last_iter, status.mals_rank_restriction, status.eta, False)
            except RuntimeError as e:
                if not _expected_newton_fallback(e):
                    raise
                Delta_tt_cc = None
            if Delta_tt_cc is not None:
                status.mals_delta0 = Delta_tt_cc
                Delta_X_tt_cc = _tt_symmetrise(tt_reshape(_tt_get_block(1, Delta_tt_cc), (2, 2)), status.eps)
                Delta_Z_tt_cc = _tt_symmetrise(tt_reshape(_tt_get_block(2, Delta_tt_cc), (2, 2)), status.eps)
                Delta_Y_tt_cc = tt_rank_reduce(_tt_get_block(0, Delta_tt_cc), eps=status.eps)
                Delta_X_tt = tt_rank_reduce(tt_add(Delta_X_tt_cc, Delta_X_tt), eps=status.eps)
                Delta_Y_tt = tt_rank_reduce(tt_add(Delta_Y_tt_cc, Delta_Y_tt), eps=status.eps)
                Delta_Z_tt = tt_rank_reduce(tt_add(Delta_Z_tt_cc, Delta_Z_tt), eps=status.eps)
                if status.ineq_status is IneqStatus.ACTIVE:
                    Delta_T_tt_cc = tt_rank_reduce(_tt_get_block(3, Delta_tt_cc), eps=status.eps)
                    Delta_T_tt_cc = tt_fast_hadamard(ineq_mask, tt_reshape(Delta_T_tt_cc, (2, 2)), status.eps)
                    Delta_T_tt = tt_rank_reduce(tt_add(Delta_T_tt_cc, Delta_T_tt), eps=status.eps)

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
        else:
            status.sigma = 0
    except Exception as e:
        status.mals_delta0 = None
        if not _expected_newton_fallback(e):
            print(f"\n\tAttention: {e}")
            print("\n\t==> Full traceback (most recent call last):")
            traceback.print_exc(file=sys.stdout)
        return 0, 0, None, None, None, None, status

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

    x_step_size, status.eigen_x0 = tt_max_generalised_eigen(X_tt, Delta_X_tt, x0=status.eigen_x0, tol=1e-8, verbose=status.verbose)
    z_step_size, status.eigen_z0 = tt_max_generalised_eigen(Z_tt, Delta_Z_tt, x0=status.eigen_z0, tol=1e-8, verbose=status.verbose)
    if status.ineq_status is not IneqStatus.NOT_IN_USE:
        if status.is_last_iter:
            X_tt = tt_add(X_tt, tt_scale(status.ineq_boundary_val + status.boundary_val, ineq_mask))
            T_tt = tt_add(T_tt, tt_scale(status.ineq_boundary_val + status.boundary_val, ineq_mask))
        x_step_size, z_step_size = _tt_get_ineq_step_sizes(x_step_size, z_step_size, X_tt, T_tt, Delta_X_tt, Delta_T_tt, ineq_mask, status)
    tau_x = 0.9 + 0.05*min(x_step_size,  z_step_size)
    tau_z = 0.9 + 0.05*min(x_step_size,  z_step_size)

    if status.verbose:
        print(f"Step search concluded.")
        print(f"Step sizes: a_p:{x_step_size:.2e}, a_d:{z_step_size:.2e}", flush=True)
    return tau_x*x_step_size, tau_z*z_step_size


def _ineq_step_size(A_tt, Delta_tt, e_tt, status):
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

    if x_step_size > 0:
        masked_X_tt = tt_fast_hadamard(ineq_mask, X_tt, status.eps)
        masked_Delta_X_tt = tt_fast_hadamard(ineq_mask, Delta_X_tt, status.eps)
        x_ineq_step_size, status.eigen_xt0 = _ineq_step_size(
            tt_add(masked_X_tt, tt_scale(status.ineq_boundary_val, ineq_mask)),
            tt_scale(x_step_size, masked_Delta_X_tt),
            status.eigen_xt0,
            status
        )
        if not status.is_last_iter:
            if 1 - x_ineq_step_size < status.op_tol and tt_norm(T_tt) < status.op_tol:
                if status.ineq_status is IneqStatus.ACTIVE:
                    status.ineq_status = IneqStatus.SETTING_INACTIVE
            else:
                if status.ineq_status is IneqStatus.INACTIVE:
                    status.ineq_status = IneqStatus.SETTING_ACTIVE
        x_step_size *= x_ineq_step_size

    if z_step_size > 0 and status.ineq_status is IneqStatus.ACTIVE:
        t_step_size, status.eigen_zt0 = _ineq_step_size(
            T_tt,
            tt_scale(z_step_size, Delta_T_tt),
            status.eigen_zt0,
            status
        )
        z_step_size *= t_step_size

    return x_step_size, z_step_size


def _initialise(ineq_mask, status, dim, lambdaStar, lambdaStarIneq):
    X_tt = tt_scale(lambdaStar, tt_identity(dim))
    Z_tt = tt_scale(lambdaStar, tt_identity(dim))
    Y_tt = tt_reshape(tt_zero_matrix(dim), (4, ))
    T_tt = None

    if status.ineq_status is IneqStatus.ACTIVE:
        T_tt = tt_scale(lambdaStarIneq, ineq_mask)
        # Need to initialise so it stays psd
        x_step_size, _ = tt_max_generalised_eigen(X_tt, ineq_mask, tol=1e-7, verbose=status.verbose)
        X_tt = tt_rank_reduce(
            tt_add(X_tt, tt_scale(0.1*x_step_size, ineq_mask)),
            status.rounding.primal_iterate(status)
        )

    return X_tt, Y_tt, Z_tt, T_tt


@dataclass(frozen=True)
class PreprocessInfo:
    norms_before: dict
    norms_after: dict
    ranks_before: dict
    ranks_after: dict
    condition: dict
    objective_offset: float
    constraint_scale: float
    psd_scale: float
    y_offset_tt: list
    objective_projected: bool


def _check_tt(name, train_tt, dim, shape=None):
    if train_tt is None:
        return
    if not isinstance(train_tt, list) or len(train_tt) != dim:
        raise ValueError(f"{name} must be a TT list of length {dim}.")
    for i, core in enumerate(train_tt):
        if not np.all(np.isfinite(core)):
            raise ValueError(f"{name}[{i}] contains non-finite data.")
        if shape is not None and tuple(core.shape[1:-1]) != tuple(shape):
            raise ValueError(f"{name}[{i}] has physical shape {core.shape[1:-1]}, expected {shape}.")
        if i and train_tt[i - 1].shape[-1] != core.shape[0]:
            raise ValueError(f"{name} has incompatible TT ranks at bond {i}.")
    if train_tt[0].shape[0] != 1 or train_tt[-1].shape[-1] != 1:
        raise ValueError(f"{name} must have boundary TT ranks equal to one.")


def _scale_tol(train_tt, op_tol, eps):
    return max(eps, op_tol * max(1.0, tt_norm(train_tt)))


def _sym_round_vec(matrix_vec_tt, op_tol, eps):
    matrix_tt = tt_reshape(matrix_vec_tt, (2, 2))
    tol = _scale_tol(matrix_tt, op_tol, eps)
    return tt_reshape(_tt_symmetrise(matrix_tt, tol), (4,))


def _validate_problem_data(lag_maps, obj_tt, lin_op_tt, bias_tt, ineq_mask):
    dim = len(obj_tt)
    _check_tt("obj_tt", obj_tt, dim, (4,))
    _check_tt("lin_op_tt", lin_op_tt, dim, (4, 4))
    _check_tt("bias_tt", bias_tt, dim, (4,))
    for key, value in lag_maps.items():
        _check_tt(f"lag_maps[{key}]", value, dim, (4, 4))
    if ineq_mask is not None:
        _check_tt("ineq_mask", ineq_mask, dim, (2, 2))


def _canonicalize_data(lag_maps, obj_tt, lin_op_tt, bias_tt, ineq_mask, op_tol, eps):
    if ineq_mask is not None:
        ineq_mask = _tt_symmetrise(ineq_mask, _scale_tol(ineq_mask, op_tol, eps))

    obj_tt = _sym_round_vec(obj_tt, op_tol, eps)
    bias_tt = _sym_round_vec(bias_tt, op_tol, eps)
    lag_maps = {k: tt_skew_zero_op(v, _scale_tol(v, op_tol, eps)) for k, v in lag_maps.items()}
    lin_op_tt = tt_rank_reduce(lin_op_tt, _scale_tol(lin_op_tt, op_tol, eps))
    return lag_maps, obj_tt, lin_op_tt, bias_tt, ineq_mask


def _project_objective(obj_tt, lin_op_tt, bias_tt, op_tol, eps):
    dim = len(obj_tt)
    diag_tt = tt_reshape(tt_diag(tt_diagonal(tt_reshape(obj_tt, (2, 2))), eps), (4,))
    zero_y = tt_reshape(tt_zero_matrix(dim), (4,))
    diag_norm = tt_norm(diag_tt)
    if diag_norm <= eps:
        return obj_tt, zero_y, 0.0, False

    projected = tt_rank_reduce(
        tt_fast_matrix_vec_mul(tt_transpose(lin_op_tt), diag_tt, eps),
        _scale_tol(diag_tt, op_tol, eps)
    )
    rel_err = tt_norm(tt_sub(projected, diag_tt)) / max(diag_norm, eps)
    if rel_err > 1e-7:
        return obj_tt, zero_y, 0.0, False

    obj_tt = tt_rank_reduce(tt_sub(obj_tt, projected), _scale_tol(obj_tt, op_tol, eps))
    return obj_tt, diag_tt, float(tt_inner_prod(bias_tt, diag_tt)), True


def _equilibrate_data(obj_tt, lin_op_tt, bias_tt, op_tol, eps):
    a_norm = max(tt_norm(lin_op_tt), eps)
    b_norm = max(tt_norm(bias_tt), eps)
    c_norm = max(tt_norm(obj_tt), eps)
    dy = float(np.clip(c_norm / a_norm, 1e-3, 1e3))
    p2 = float(np.clip(b_norm / a_norm, 1e-3, 1e3))

    obj_tt = tt_scale(p2, obj_tt)
    lin_op_tt = tt_scale(dy * p2, lin_op_tt)
    bias_tt = tt_scale(dy, bias_tt)
    condition = {
        "lin_op_norm": a_norm,
        "bias_norm": b_norm,
        "objective_norm": c_norm,
        "bias_to_operator": b_norm / a_norm,
        "objective_to_operator": c_norm / a_norm,
    }
    return obj_tt, lin_op_tt, bias_tt, dy, np.sqrt(p2), condition


def _auto_initial_scales(preprocess, dim, ineq_active, lambdaStar, lambdaStarIneq):
    if lambdaStar is None:
        b = max(preprocess.norms_after["bias"], 1.0)
        c = max(preprocess.norms_after["objective"], 1.0)
        scale = np.sqrt(1.0 + max(b, c))
        lambdaStar = float(np.clip(2.0 ** np.round(np.log2(scale)), 1e-6, 2.0))
    if lambdaStarIneq is None:
        mask = max(preprocess.norms_after.get("ineq_mask", 1.0), 1.0)
        c = max(preprocess.norms_after["objective"], 1.0)
        lambdaStarIneq = float(np.clip(c / mask, 1e-8, 1e6)) if ineq_active else 1.0
    return lambdaStar, lambdaStarIneq


def _preprocess_problem_data(lag_maps, obj_tt, lin_op_tt, bias_tt, ineq_mask, op_tol, eps):
    _validate_problem_data(lag_maps, obj_tt, lin_op_tt, bias_tt, ineq_mask)
    names = ("objective", "operator", "bias", "ineq_mask")
    tensors_before = (obj_tt, lin_op_tt, bias_tt, ineq_mask)
    norms_before = {n: (tt_norm(t) if t is not None else 0.0) for n, t in zip(names, tensors_before)}
    ranks_before = {n: (tt_ranks(t) if t is not None else []) for n, t in zip(names, tensors_before)}

    lag_maps, obj_tt, lin_op_tt, bias_tt, ineq_mask = _canonicalize_data(
        lag_maps, obj_tt, lin_op_tt, bias_tt, ineq_mask, op_tol, eps
    )
    obj_tt, y_offset, objective_offset, projected = _project_objective(obj_tt, lin_op_tt, bias_tt, op_tol, eps)
    obj_tt, lin_op_tt, bias_tt, dy, p, condition = _equilibrate_data(obj_tt, lin_op_tt, bias_tt, op_tol, eps)

    tensors_after = (obj_tt, lin_op_tt, bias_tt, ineq_mask)
    norms_after = {n: (tt_norm(t) if t is not None else 0.0) for n, t in zip(names, tensors_after)}
    ranks_after = {n: (tt_ranks(t) if t is not None else []) for n, t in zip(names, tensors_after)}
    return lag_maps, obj_tt, lin_op_tt, bias_tt, ineq_mask, PreprocessInfo(
        norms_before, norms_after, ranks_before, ranks_after, condition,
        objective_offset, dy, p, y_offset, projected
    )


def _unscale_output(X_tt, Y_tt, T_tt, Z_tt, status):
    p2 = status.preprocess.psd_scale ** 2
    X_tt = tt_rank_reduce(tt_scale(p2, X_tt), status.eps)
    Z_tt = tt_rank_reduce(tt_scale(1.0 / p2, Z_tt), status.eps)
    Y_tt = tt_rank_reduce(
        tt_add(tt_scale(status.preprocess.constraint_scale, Y_tt), status.preprocess.y_offset_tt),
        status.eps
    )
    if T_tt is not None:
        T_tt = tt_rank_reduce(tt_scale(1.0 / p2, T_tt), status.eps)
    return X_tt, Y_tt, T_tt, Z_tt


@dataclass(frozen=True)
class RoundingPolicy:
    eta_floor: float
    residual_factor: float = 0.01
    iterate_factor: float = 1.0
    kkt_factor: float = 1.0

    def _kkt_split(self, status):
        return np.sqrt(4 if status.ineq_status is IneqStatus.ACTIVE else 3)

    def primal_residual(self, status):
        return self.residual_factor * status.eta * status.primal_error_normalisation

    def dual_residual(self, status):
        return self.residual_factor * status.eta * status.dual_error_normalisation

    def centrality_residual(self, status):
        return self.residual_factor * status.eta * status.centrl_error_normalisation

    def primal_iterate(self, status):
        return self.iterate_factor * status.eta * status.primal_error_normalisation

    def dual_iterate(self, status):
        return self.iterate_factor * status.eta * status.dual_error_normalisation

    def kkt_primal(self, status):
        return self.kkt_factor * status.eta * status.primal_error_normalisation / self._kkt_split(status)

    def kkt_dual(self, status):
        return self.kkt_factor * status.eta * status.dual_error_normalisation / self._kkt_split(status)


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
    rounding: RoundingPolicy

    boundary_val: float = 1e-10
    ineq_boundary_val: float = 0.01
    sigma: float = 0.5
    num_ineq_constraints: float = 0
    lag_map_t = None
    lag_map_y = None
    compl_ineq_mask = None
    mals_delta0 = None
    eigen_x0 = None
    eigen_z0 = None
    eigen_xt0 = None
    eigen_zt0 = None
    kkt_iterations = 7
    centrl_error_normalisation: float = 1.0
    eta = 1e-3
    preprocess: PreprocessInfo = None
    converged: bool = False


def _ipm_format_output(X_tt, Y_tt, T_tt, Z_tt, iteration, status):
    """Formats the final results into the desired output structure."""
    ranksX = tt_ranks(X_tt)
    ranksZ = tt_ranks(Z_tt)
    ranksY = tt_ranks(Y_tt)
    ranksT = tt_ranks(T_tt) if T_tt else [0] * (status.dim - 1)
    
    print("---Terminated---")
    verb = "Converged" if status.converged else "Stopped"
    print(f"{verb} in {iteration} iterations.")
    print(f"Ranks: X={ranksX}, Z={ranksZ}, Y={ranksY}, T={ranksT}")
    
    results = {"num_iters": iteration, "ranksX": ranksX, "ranksY": ranksY, "ranksZ": ranksZ, "ranksT": ranksT, "status": status}
    return X_tt, Y_tt, T_tt, Z_tt, results


def _ipm_check_for_stalled_progress(prev_errors, status, gap_tol):
    """Checks if the optimization has stalled."""
    if status.is_last_iter:
        return False
    if not _ipm_ready_for_refinement(status):
        return False
        
    primal_stalled = abs(prev_errors['primal'] - status.primal_error) < 0.04 * gap_tol
    dual_stalled = abs(prev_errors['dual'] - status.dual_error) < 0.04 * gap_tol
    centrality_stalled = abs(prev_errors['centrality'] - status.centrality_error) < 0.02 * gap_tol
    
    if primal_stalled and dual_stalled and centrality_stalled:
        if status.verbose:
            print("============================================\n Progress stalled! Entering finishing phase.\n============================================")
        return True
    return False


def _ipm_ready_for_refinement(status):
    return (
        status.is_primal_feasible
        and status.is_dual_feasible
        and status.centrality_error < status.centrality_tol
    )


def _ipm_check_convergence(status, finishing_steps, ZX, TX, abs_tol, max_refinement):
    """Checks for final convergence and updates the finishing step counter."""
    if not status.is_last_iter:
        return status, finishing_steps
        
    converged = (abs(ZX) + abs(TX) < abs_tol and 
                 status.primal_error < abs_tol and 
                 status.dual_error < abs_tol)

    if converged:
        if status.verbose:
            print("Absolute tolerance reached!")
        finishing_steps = 0
        status.converged = True
    else:
        finishing_steps -= 1
        status.boundary_val = 0.001 * (1 - (finishing_steps / max_refinement))
        if finishing_steps == 1:
            status.kkt_iterations += 1
            
    return status, finishing_steps


def _ipm_log_iteration(iteration, status, X_tt, Y_tt, Z_tt, T_tt):
    """Prints verbose output for the current iteration."""
    print(f"\n--- Iteration {iteration - 1} ---")
    print(f"Status: Finishing up={status.is_last_iter}, Ineq={str(status.ineq_status)}")
    print(f"Feasibility: Central={status.is_central}, Primal={status.is_primal_feasible}, Dual={status.is_dual_feasible}")
    print(f"Direction: {'AHO' if status.aho_direction else 'XZ'}, Sigma: {status.sigma:.2e}")
    print(f"Errors: Centrality={status.centrality_error:.4e}, Primal={status.primal_error:.4e}, Dual={status.dual_error:.4e}")
    print(f"Ranks: X={tt_ranks(X_tt)}, Z={tt_ranks(Z_tt)}, Y={tt_ranks(Y_tt)}, T={tt_ranks(T_tt) if T_tt else 'N/A'}", flush=True)


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
    lambdaStar=None,
    lambdaStarIneq=None,
    eta_floor=None,
    epsilonDash=None,
    epsilonDashineq=None,
    verbose=False
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
        verbose,
        1,
        1,
        r_max,
        RoundingPolicy(op_tol if eta_floor is None else eta_floor)
    )
    lag_maps, obj_tt, lin_op_tt, bias_tt, ineq_mask, status.preprocess = _preprocess_problem_data(
        lag_maps, obj_tt, lin_op_tt, bias_tt, ineq_mask, op_tol, eps
    )
    lambdaStar, lambdaStarIneq = _auto_initial_scales(
        status.preprocess, dim, status.ineq_status is IneqStatus.ACTIVE, lambdaStar, lambdaStarIneq
    )

    status.primal_error_normalisation = 1 + tt_norm(bias_tt)
    status.dual_error_normalisation = 1 + tt_norm(obj_tt)

    lhs_skeleton = TTBlockMatrix()
    lhs_skeleton[1, 2] = tt_reshape(tt_identity(2 * dim), (4, 4))
    solver_ineq = lambda lhs, rhs, x0, nwsp, restriction, termination_tol, strict_forcing=True: tt_restarted_block_amen(
        lhs,
        rhs,
        rank_restriction=restriction,
        x0=x0,
        local_solver=_ipm_local_solver_ineq,
        op_tol=op_tol,
        termination_tol=termination_tol,
        num_restarts=mals_restarts,
        inner_m=nwsp,
        verbose=verbose,
        strict_forcing=strict_forcing
    )
    solver_eq = lambda lhs, rhs, x0, nwsp, restriction, termination_tol, strict_forcing=True: tt_restarted_block_amen(
        lhs,
        rhs,
        rank_restriction=restriction,
        x0=x0,
        local_solver=_ipm_local_solver,
        op_tol=op_tol,
        termination_tol=termination_tol,
        num_restarts=mals_restarts, 
        inner_m=nwsp,
        verbose=verbose,
        strict_forcing=strict_forcing
    )
    if status.ineq_status is IneqStatus.ACTIVE:
        solver = solver_ineq
        status.num_ineq_constraints = tt_inner_prod(ineq_mask, ineq_mask)
        status.compl_ineq_mask = tt_rank_reduce(tt_sub(tt_one_matrix(dim), ineq_mask), eps=eps)
        status.lag_map_t = lag_maps["t"]
        lhs_skeleton.add_alias((1, 2), (1, 3))
    else:
        solver = solver_eq
        status.num_ineq_constraints = 0

    # KKT-system prep
    lin_op_tt_adj = tt_transpose(lin_op_tt)
    lhs_skeleton[0, 1] = tt_scale(-1, lin_op_tt)
    lhs_skeleton.add_alias((0, 1), (1, 0), is_transpose=True)
    lhs_skeleton[0, 0] = lag_maps["y"]
    status.lag_map_y = lag_maps["y"]

    X_tt, Y_tt, Z_tt, T_tt = _initialise(ineq_mask, status, dim, lambdaStar, lambdaStarIneq)

    iteration = 0
    finishing_steps = max_refinement
    prev_errors = {'primal': np.inf, 'dual': np.inf, 'centrality': np.inf}
    lhs = lhs_skeleton

    while finishing_steps > 0:
        iteration += 1
        status.aho_direction = (iteration > warm_up)
        ready_to_refine = _ipm_ready_for_refinement(status)
        if not status.is_last_iter and (
            (max_iter - max_refinement == iteration - 1 and ready_to_refine)
            or iteration - 1 >= max_iter
        ):
            if status.verbose:
                msg = "Entering final refinement phase."
                if iteration - 1 >= max_iter:
                    msg = "Max iterations reached. Entering final refinement phase."
                print(f"\n{msg}")
            status.is_last_iter = True
        ZX = tt_inner_prod(Z_tt, X_tt)
        TX = tt_inner_prod(X_tt, T_tt) + status.ineq_boundary_val*tt_entrywise_sum(T_tt) if status.ineq_status is IneqStatus.ACTIVE else 0
        status.mu = np.divide(abs(ZX) + abs(TX), (2 ** dim + (status.ineq_status is IneqStatus.ACTIVE)*status.num_ineq_constraints))
        status.centrl_error_normalisation = 1 + abs(tt_inner_prod(obj_tt, tt_reshape(X_tt, (4, ))))
        complementarity = abs(ZX) + abs(TX)
        status.centrality_error = status.mu / status.centrl_error_normalisation
        status.is_central = status.centrality_error < centrality_tol and complementarity < 0.5 * abs_tol
        status.eta = max(min(status.eta, 2*status.mu), status.rounding.eta_floor)

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

        if verbose:
            _ipm_log_iteration(iteration, status, X_tt, Y_tt, Z_tt, T_tt)

        status, finishing_steps = _ipm_check_convergence(
            status, finishing_steps, ZX, TX, abs_tol, max_refinement
        )
        if finishing_steps == 0:
            iteration -= 1
            break

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

        if (Delta_X_tt is None and Delta_Z_tt is None) or (x_step_size < 1e-5 and z_step_size < 1e-5):
            if abs(ZX) + abs(TX) < abs_tol and status.primal_error < abs_tol and status.dual_error < abs_tol:
                status.converged = True
                break
            if status.is_last_iter:
                break
            else:
                if status.verbose:
                    print("\nEntering final refinement phase.")
                status.is_last_iter = True
        else:
            if finishing_steps <= 1:
                X_tt = _tt_symmetrise(
                    tt_add(X_tt, tt_scale(x_step_size, Delta_X_tt)),
                    status.rounding.primal_iterate(status)
                )
            else:
                X_tt = _tt_psd_symmetrise(
                    tt_add(X_tt, tt_scale(x_step_size, Delta_X_tt)),
                    status.rounding.primal_iterate(status)
                )
            if finishing_steps <= 1:
                Z_tt = _tt_symmetrise(
                    tt_add(Z_tt, tt_scale(z_step_size, Delta_Z_tt)),
                    status.rounding.dual_iterate(status)
                )
            else:
                Z_tt = _tt_psd_symmetrise(
                    tt_add(Z_tt, tt_scale(z_step_size, Delta_Z_tt)),
                    status.rounding.dual_iterate(status)
                )

            Y_tt = tt_rank_reduce(
                tt_add(Y_tt, tt_scale(z_step_size, Delta_Y_tt)),
                status.rounding.dual_iterate(status)
            )
            Y_tt = tt_reshape(
                _tt_symmetrise(
                    tt_reshape(tt_sub(Y_tt, tt_fast_matrix_vec_mul(status.lag_map_y, Y_tt, status.eps)), (2, 2)),
                    status.rounding.dual_iterate(status)
                ),
                (4, )
            )

            if status.ineq_status is IneqStatus.ACTIVE:
                if finishing_steps <= 1:
                    T_tt = _tt_symmetrise(
                        tt_add(T_tt, tt_scale(z_step_size, Delta_T_tt)),
                        status.rounding.dual_iterate(status)
                    )
                else:
                    T_tt = _tt_mask_symmetrise(
                        tt_add(T_tt, tt_scale(z_step_size, Delta_T_tt)),
                        ineq_mask,
                        status.rounding.dual_iterate(status)
                    )
            elif status.ineq_status is IneqStatus.SETTING_INACTIVE:
                solver = solver_eq
                lhs = lhs_skeleton.get_submatrix(2, 2)
                status.mals_delta0 = None
                status.ineq_status = IneqStatus.INACTIVE
            elif status.ineq_status is IneqStatus.SETTING_ACTIVE:
                solver = solver_ineq
                lhs = lhs_skeleton
                status.mals_delta0 = None
                status.ineq_status = IneqStatus.ACTIVE

        if _ipm_check_for_stalled_progress(prev_errors, status, gap_tol):
            status.is_last_iter = True

        prev_errors['primal'] = status.primal_error
        prev_errors['dual'] = status.dual_error
        prev_errors['centrality'] = status.centrality_error

    X_tt, Y_tt, T_tt, Z_tt = _unscale_output(X_tt, Y_tt, T_tt, Z_tt, status)
    return _ipm_format_output(X_tt, Y_tt, T_tt, Z_tt, iteration, status)
