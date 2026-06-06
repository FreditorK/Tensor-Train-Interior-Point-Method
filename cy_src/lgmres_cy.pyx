# lgmres_cy.pyx
# cython: language_level=3
# cython: cdivision=True
# cython: optimize.use_switch=True
# distutils: language = c++

cdef extern from "numpy/arrayobject.h":
    # Define NPY_NO_DEPRECATED_API for compatibility with numpy
    ctypedef void npy_no_deprecated_api

import numpy as np
cimport numpy as cnp
cimport cython
cimport scipy.linalg.cython_blas as blas
from libc.string cimport memcpy
from scipy.linalg.cython_blas cimport dcopy

cnp.import_array() # Initialize NumPy C-API

cdef:
    int inc = 1  # typical unit stride
    double global_alpha = 1.0

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.inline
cdef void cy_maxpy(double[:, :] dx, double[:, :] x) noexcept nogil:
    cdef int n = x.shape[0] * x.shape[1]
    blas.daxpy(&n, &global_alpha, &dx[0, 0], &inc, &x[0, 0], &inc)

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.inline
cdef void cy_dgemm(
        const double[:, ::1] A,
        const double[:, ::1] B,
        double[:, :] C,
        double alpha=1.0,
        double beta=0.0
) noexcept nogil:
    cdef int M = A.shape[0]
    cdef int K = A.shape[1]
    cdef int N = B.shape[1]
    cdef char trans = 78  # ord("N")
    # Row-major C = A B is equivalent to column-major C.T = B.T A.T.
    blas.dgemm(&trans, &trans, &N, &M, &K, &alpha,
               <double*>&B[0, 0], &N, <double*>&A[0, 0], &K, &beta, &C[0, 0], &N)


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.inline
cdef void cy_dgemv_row(
        const double[:, ::1] A,
        const double[::1] x,
        double[::1] y,
        double alpha=1.0,
        double beta=0.0
) noexcept nogil:
    cdef int M = A.shape[0]
    cdef int N = A.shape[1]
    cdef char trans = 84  # ord("T")
    # Row-major y = A x is column-major dgemv on A.T.
    blas.dgemv(&trans, &N, &M, &alpha,
               <double*>&A[0, 0], &N, <double*>&x[0], &inc, &beta, &y[0], &inc)


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.inline
cdef void _transpose_reshape_step2(
    const double[:, ::1] src_2d, # Input: (r* n, R*S)
    double[:, ::1] dest, # Output: (r * R, n * S)
    int r, 
    int R, 
    int n, 
    int S
) noexcept nogil:
    cdef int i, j, k
    cdef size_t block_size = S * sizeof(double)
    
    for i in range(r):
        for j in range(R):
            for k in range(n):
                memcpy(
                    &dest[i * R + j, k * S], 
                    &src_2d[i * n + k, j * S],
                    block_size
                )


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.inline
cdef void _transpose_reshape_step3(
    const double[:, ::1] src_2d,
    double[:, ::1] dest,
    int r, int R, int s, int n
) noexcept nogil:
    cdef int i, j, l
    cdef int n_copy = s
    cdef int incx = n
    cdef int incy = 1

    for i in range(r):
        for j in range(R):
            for l in range(n):
                dcopy(
                    &n_copy,
                    <double*>&src_2d[i * R + j, l],
                    &incx,
                    &dest[j * n + l, i * s],
                    &incy
                )


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.inline
cdef void _transpose_reshape_multiply_inplace(
    const double[:, ::1] src,      # shape (R*n, r)
    double[:, ::1] dest,          # shape (r*n, R)
    const double[:, ::1] source_to_multiply, # The array to multiply by
    int r,
    int n,
    int R
) noexcept nogil:
    cdef int i, j, k
    
    for i in range(r):
        for j in range(n):
            for k in range(R):
                dest[i * n + j, k] = src[k * n + j, i] * source_to_multiply[i * n + j, k]


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
cdef void einsum(
        const double[:, ::1] XAX,
        const double[:, ::1] block_A,
        const double[:, ::1] XAX1,
        const double[:, ::1] x_core,
        double[:, ::1] out,
        double[:, ::1] intermediate_mat1,
        double[:, ::1] intermediate_mat1_2,
        double[:, ::1] intermediate_mat2,
        double[:, ::1] intermediate_mat2_2,
        int r,
        int n,
        int R,
        double alpha,
        double beta
) noexcept nogil:
    cdef:
        int S = int(block_A.shape[0] / n)
        int s = int(block_A.shape[1] / n)

    # einsum: rnR,LSR -> rnLS
    cy_dgemm(x_core, XAX1, intermediate_mat1) # rn x RS # x_core does not need to be contigous here as via reshape still same memory layout
    # einsum: rnLS,smnS -> rLsm
    _transpose_reshape_step2(intermediate_mat1, intermediate_mat1_2, r, R, n, S)
    cy_dgemm(intermediate_mat1_2, block_A, intermediate_mat2)
    # einsum: rLsm,lsr -> Lml
    _transpose_reshape_step3(intermediate_mat2, intermediate_mat2_2, r, R, s, n)
    cy_dgemm(intermediate_mat2_2, XAX, out, alpha, beta)


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.inline
cdef void pack_results(double[:, :] result0, double[:, :] result1, double[:] flat_result, int R, int n, int r) noexcept nogil:
    cdef int i, j, k
    cdef int idx

    for i in range(r):
        for j in range(n):
            for k in range(R):
                idx = (i * n + j) * R + k
                flat_result[idx] = result0[k * n + j, i]

    for i in range(r):
        for j in range(n):
            for k in range(R):
                idx = ((r + i) * n + j) * R + k
                flat_result[idx] = result1[k * n + j, i]


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.inline
cdef void pack_results3(double[:, :] result0, double[:, :] result1, double[:, :] result2, double[:] flat_result, int R, int n, int r) noexcept nogil:
    cdef int i, j, k
    cdef int idx
    cdef int base = r * n * R

    for i in range(r):
        for j in range(n):
            for k in range(R):
                idx = (i * n + j) * R + k
                flat_result[idx] = result0[k * n + j, i]
                flat_result[base + idx] = result1[k * n + j, i]
                flat_result[2 * base + idx] = result2[k * n + j, i]


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.inline
cdef void pack_result(double[:, :] result, double[:] flat_result, int R, int n, int r) noexcept nogil:
    cdef int i, j, k
    cdef int idx

    for i in range(r):
        for j in range(n):
            for k in range(R):
                idx = (i * n + j) * R + k
                flat_result[idx] = result[k * n + j, i]


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.inline
cdef void pack_result_shift(
        double[:, :] result,
        double[:] flat_result,
        const double[:] x,
        int R,
        int n,
        int r,
        double shift
) noexcept nogil:
    cdef int i, j, k
    cdef int idx

    for i in range(r):
        for j in range(n):
            for k in range(R):
                idx = (i * n + j) * R + k
                flat_result[idx] = result[k * n + j, i] + shift * x[idx]


cdef class BaseMatVec:

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.nonecheck(False)
    cpdef cnp.ndarray[double, ndim=1] matvec(self, cnp.ndarray[double, ndim=1] x_core):
        raise NotImplementedError("BaseMatVec.matvec must be implemented by subclass")

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.nonecheck(False)
    cpdef matvec_into(self, cnp.ndarray[double, ndim=1] x_core, cnp.ndarray[double, ndim=1] out):
        raise NotImplementedError("BaseMatVec.matvec_into must be implemented by subclass")


cdef class DiagTwoCoreBlockWrapper:
    cdef const double[:, :, ::1] XAX, A0, A1, XAX2
    cdef const double[:, :, ::1] XAX_by_r, A0_by_n, A1_by_p, XAX2_by_R_T
    cdef double[:, ::1] coeff, tmp, small
    cdef int ldim, sdim, rdim, n0, kdim, n1, Sdim, Ldim, Rdim
    cdef int out_size, in_size

    def __init__(
            self,
            cnp.ndarray[double, ndim=3] XAX,
            cnp.ndarray[double, ndim=3] A0,
            cnp.ndarray[double, ndim=3] A1,
            cnp.ndarray[double, ndim=3] XAX2
    ):
        self.XAX = np.ascontiguousarray(XAX)
        self.A0 = np.ascontiguousarray(A0)
        self.A1 = np.ascontiguousarray(A1)
        self.XAX2 = np.ascontiguousarray(XAX2)
        self.XAX_by_r = np.ascontiguousarray(np.transpose(XAX, (2, 0, 1)))
        self.A0_by_n = np.ascontiguousarray(np.transpose(A0, (1, 0, 2)))
        self.A1_by_p = np.ascontiguousarray(np.transpose(A1, (1, 0, 2)))
        self.XAX2_by_R_T = np.ascontiguousarray(np.transpose(XAX2, (2, 1, 0)))
        self.ldim = XAX.shape[0]
        self.sdim = XAX.shape[1]
        self.rdim = XAX.shape[2]
        self.n0 = A0.shape[1]
        self.kdim = A0.shape[2]
        self.n1 = A1.shape[1]
        self.Sdim = A1.shape[2]
        self.Ldim = XAX2.shape[0]
        self.Rdim = XAX2.shape[2]
        self.out_size = self.ldim * self.n0 * self.n1 * self.Ldim
        self.in_size = self.rdim * self.n0 * self.n1 * self.Rdim
        self.coeff = np.empty((self.sdim, self.Sdim), dtype=np.float64)
        self.tmp = np.empty((self.ldim, self.Sdim), dtype=np.float64)
        self.small = np.empty((self.ldim, self.Ldim), dtype=np.float64)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.nonecheck(False)
    cpdef cnp.ndarray[double, ndim=2] block(self, int n, int p):
        cdef cnp.ndarray[double, ndim=2] out_arr = np.empty(
            (self.ldim * self.Ldim, self.rdim * self.Rdim), dtype=np.float64
        )
        cdef double[:, ::1] out = out_arr
        cdef const double[:, :, ::1] XAX_by_r = self.XAX_by_r
        cdef const double[:, :, ::1] A0_by_n = self.A0_by_n
        cdef const double[:, :, ::1] A1_by_p = self.A1_by_p
        cdef const double[:, :, ::1] XAX2_by_R_T = self.XAX2_by_R_T
        cdef double[:, ::1] coeff = self.coeff
        cdef double[:, ::1] tmp = self.tmp
        cdef double[:, ::1] small = self.small
        cdef int li, Li, ri, Ri
        with nogil:
            cy_dgemm(A0_by_n[n, :, :], A1_by_p[p, :, :], coeff)
            for ri in range(self.rdim):
                cy_dgemm(XAX_by_r[ri, :, :], coeff, tmp)
                for Ri in range(self.Rdim):
                    cy_dgemm(tmp, XAX2_by_R_T[Ri, :, :], small)
                    for li in range(self.ldim):
                        for Li in range(self.Ldim):
                            out[li * self.Ldim + Li, ri * self.Rdim + Ri] = small[li, Li]
        return out_arr

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.nonecheck(False)
    cpdef cnp.ndarray[double, ndim=1] matvec(self, cnp.ndarray[double, ndim=1] x_vec):
        cdef cnp.ndarray[double, ndim=1] x_arr = np.ascontiguousarray(x_vec, dtype=np.float64)
        cdef cnp.ndarray[double, ndim=1] out_arr = np.zeros(self.out_size, dtype=np.float64)
        cdef const double[::1] x = x_arr
        cdef double[::1] out = out_arr
        cdef cnp.ndarray[double, ndim=1] x_block_arr = np.empty(self.rdim * self.Rdim, dtype=np.float64)
        cdef cnp.ndarray[double, ndim=1] y_block_arr = np.empty(self.ldim * self.Ldim, dtype=np.float64)
        cdef double[::1] x_block = x_block_arr
        cdef double[::1] y_block = y_block_arr
        cdef cnp.ndarray[double, ndim=2] block_arr
        cdef double[:, ::1] block_view
        cdef int li, Li, ri, Ri, n, p
        for n in range(self.n0):
            for p in range(self.n1):
                for ri in range(self.rdim):
                    for Ri in range(self.Rdim):
                        x_block[ri * self.Rdim + Ri] = x[((ri * self.n0 + n) * self.n1 + p) * self.Rdim + Ri]
                block_arr = self.block(n, p)
                block_view = block_arr
                with nogil:
                    cy_dgemv_row(block_view, x_block, y_block)
                    for li in range(self.ldim):
                        for Li in range(self.Ldim):
                            out[((li * self.n0 + n) * self.n1 + p) * self.Ldim + Li] = y_block[li * self.Ldim + Li]
        return out_arr


cdef class DiagOneCoreBlockWrapper:
    cdef const double[:, :, ::1] XAX, A0, XAX1
    cdef const double[:, :, ::1] XAX_by_r, A0_by_n, XAX1_by_R_T
    cdef double[:, ::1] tmp, small
    cdef int ldim, sdim, rdim, n0, Sdim, Ldim, Rdim
    cdef int out_size, in_size

    def __init__(
            self,
            cnp.ndarray[double, ndim=3] XAX,
            cnp.ndarray[double, ndim=3] A0,
            cnp.ndarray[double, ndim=3] XAX1
    ):
        self.XAX = np.ascontiguousarray(XAX)
        self.A0 = np.ascontiguousarray(A0)
        self.XAX1 = np.ascontiguousarray(XAX1)
        self.XAX_by_r = np.ascontiguousarray(np.transpose(XAX, (2, 0, 1)))
        self.A0_by_n = np.ascontiguousarray(np.transpose(A0, (1, 0, 2)))
        self.XAX1_by_R_T = np.ascontiguousarray(np.transpose(XAX1, (2, 1, 0)))
        self.ldim = XAX.shape[0]
        self.sdim = XAX.shape[1]
        self.rdim = XAX.shape[2]
        self.n0 = A0.shape[1]
        self.Sdim = A0.shape[2]
        self.Ldim = XAX1.shape[0]
        self.Rdim = XAX1.shape[2]
        self.out_size = self.ldim * self.n0 * self.Ldim
        self.in_size = self.rdim * self.n0 * self.Rdim
        self.tmp = np.empty((self.ldim, self.Sdim), dtype=np.float64)
        self.small = np.empty((self.ldim, self.Ldim), dtype=np.float64)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.nonecheck(False)
    cpdef cnp.ndarray[double, ndim=2] block(self, int n):
        cdef cnp.ndarray[double, ndim=2] out_arr = np.empty(
            (self.ldim * self.Ldim, self.rdim * self.Rdim), dtype=np.float64
        )
        cdef double[:, ::1] out = out_arr
        cdef const double[:, :, ::1] XAX_by_r = self.XAX_by_r
        cdef const double[:, :, ::1] A0_by_n = self.A0_by_n
        cdef const double[:, :, ::1] XAX1_by_R_T = self.XAX1_by_R_T
        cdef double[:, ::1] tmp = self.tmp
        cdef double[:, ::1] small = self.small
        cdef int li, Li, ri, Ri
        with nogil:
            for ri in range(self.rdim):
                cy_dgemm(XAX_by_r[ri, :, :], A0_by_n[n, :, :], tmp)
                for Ri in range(self.Rdim):
                    cy_dgemm(tmp, XAX1_by_R_T[Ri, :, :], small)
                    for li in range(self.ldim):
                        for Li in range(self.Ldim):
                            out[li * self.Ldim + Li, ri * self.Rdim + Ri] = small[li, Li]
        return out_arr

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.nonecheck(False)
    cpdef cnp.ndarray[double, ndim=1] matvec(self, cnp.ndarray[double, ndim=1] x_vec):
        cdef cnp.ndarray[double, ndim=1] x_arr = np.ascontiguousarray(x_vec, dtype=np.float64)
        cdef cnp.ndarray[double, ndim=1] out_arr = np.zeros(self.out_size, dtype=np.float64)
        cdef const double[::1] x = x_arr
        cdef double[::1] out = out_arr
        cdef cnp.ndarray[double, ndim=1] x_block_arr = np.empty(self.rdim * self.Rdim, dtype=np.float64)
        cdef cnp.ndarray[double, ndim=1] y_block_arr = np.empty(self.ldim * self.Ldim, dtype=np.float64)
        cdef double[::1] x_block = x_block_arr
        cdef double[::1] y_block = y_block_arr
        cdef cnp.ndarray[double, ndim=2] block_arr
        cdef double[:, ::1] block_view
        cdef int li, Li, ri, Ri, n
        for n in range(self.n0):
            for ri in range(self.rdim):
                for Ri in range(self.Rdim):
                    x_block[ri * self.Rdim + Ri] = x[(ri * self.n0 + n) * self.Rdim + Ri]
            block_arr = self.block(n)
            block_view = block_arr
            with nogil:
                cy_dgemv_row(block_view, x_block, y_block)
                for li in range(self.ldim):
                    for Li in range(self.Ldim):
                        out[(li * self.n0 + n) * self.Ldim + Li] = y_block[li * self.Ldim + Li]
        return out_arr


cdef class SymOneCoreMatVecWrapper:
    cdef double[:, ::1] result
    cdef object flat_result_arr
    cdef double[:] flat_result
    cdef const double[:, ::1] XAX, XAX_T, block_A, block_A_T, XAX1, XAX1_T
    cdef double[:, ::1] workspace1, workspace1_2, workspace2, workspace2_2
    cdef int r, n, R, total_size
    cdef double diagonal_shift

    def __init__(
            self,
            cnp.ndarray[double, ndim=3] Phi_l,
            cnp.ndarray[double, ndim=4] A_k,
            cnp.ndarray[double, ndim=3] Phi_r,
            int r,
            int n,
            int R,
            double diagonal_shift=0.0
    ):
        cdef object Phi_l_T = np.transpose(Phi_l, axes=(2, 1, 0))
        cdef object A_k_T = np.transpose(A_k, axes=(0, 2, 1, 3))
        cdef object Phi_r_T = np.transpose(Phi_r, axes=(2, 1, 0))

        self.XAX = np.ascontiguousarray(Phi_l.transpose(0, 2, 1).reshape(Phi_l.shape[0], -1).T)
        self.block_A = np.ascontiguousarray(A_k.reshape(A_k.shape[0] * A_k.shape[1], A_k.shape[2] * A_k.shape[3]).T)
        self.XAX1 = np.ascontiguousarray(Phi_r.reshape(-1, R).T)
        self.XAX_T = np.ascontiguousarray(Phi_l_T.transpose(0, 2, 1).reshape(Phi_l_T.shape[0], -1).T)
        self.block_A_T = np.ascontiguousarray(A_k_T.reshape(A_k_T.shape[0] * A_k_T.shape[1], A_k_T.shape[2] * A_k_T.shape[3]).T)
        self.XAX1_T = np.ascontiguousarray(Phi_r_T.reshape(-1, R).T)

        self.r = r
        self.n = n
        self.R = R
        self.diagonal_shift = diagonal_shift
        self.total_size = r * n * R
        self.result = np.empty((R * n, r), dtype=np.float64)
        self.workspace1 = np.empty((r * n, R * A_k.shape[3]), dtype=np.float64)
        self.workspace1_2 = np.empty((r * R, n * A_k.shape[3]), dtype=np.float64)
        self.workspace2 = np.empty((r * R, A_k.shape[0] * n), dtype=np.float64)
        self.workspace2_2 = np.empty((R * n, r * A_k.shape[0]), dtype=np.float64)
        self.flat_result_arr = np.empty(self.total_size, dtype=np.float64)
        self.flat_result = self.flat_result_arr

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.nonecheck(False)
    cpdef cnp.ndarray[double, ndim=1] matvec(self, cnp.ndarray[double, ndim=1] x_core):
        cdef const double[:, ::1] x_view = x_core.reshape(self.r * self.n, self.R)
        cdef const double[:] x_flat = x_core

        with nogil:
            einsum(
                self.XAX, self.block_A, self.XAX1,
                x_view, self.result,
                self.workspace1, self.workspace1_2, self.workspace2, self.workspace2_2,
                self.r, self.n, self.R, 0.5, 0.0
            )
            einsum(
                self.XAX_T, self.block_A_T, self.XAX1_T,
                x_view, self.result,
                self.workspace1, self.workspace1_2, self.workspace2, self.workspace2_2,
                self.r, self.n, self.R, 0.5, 1.0
            )
            if self.diagonal_shift != 0.0:
                pack_result_shift(self.result, self.flat_result, x_flat, self.R, self.n, self.r, self.diagonal_shift)
            else:
                pack_result(self.result, self.flat_result, self.R, self.n, self.r)
        return self.flat_result_arr


cdef class SymTwoCoreMatVecWrapper:
    cdef object inner

    def __init__(
            self,
            cnp.ndarray[double, ndim=3] Phi_l,
            cnp.ndarray[double, ndim=4] A_l,
            cnp.ndarray[double, ndim=4] A_r,
            cnp.ndarray[double, ndim=3] Phi_r,
            int r,
            int n0,
            int n1,
            int R,
            double diagonal_shift=0.0
    ):
        cdef cnp.ndarray[double, ndim=4] A_pair = np.ascontiguousarray(
            np.einsum("smnk,kptS->smpntS", A_l, A_r, optimize=True).reshape(
                A_l.shape[0], n0 * n1, n0 * n1, A_r.shape[3]
            )
        )
        self.inner = SymOneCoreMatVecWrapper(Phi_l, A_pair, Phi_r, r, n0 * n1, R, diagonal_shift)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.nonecheck(False)
    cpdef cnp.ndarray[double, ndim=1] matvec(self, cnp.ndarray[double, ndim=1] x_core):
        return self.inner.matvec(x_core)


cdef class MatVecWrapper(BaseMatVec):
    cdef double[:,  ::1] result0, result1, temp, x_reshaped_0, x_reshaped_1
    cdef object flat_result_arr
    cdef double[:] flat_result
    cdef const double[:,  ::1] XAX_k_00, XAX_k_01, XAX_k_01T, XAX_k_21, XAX_k_22
    cdef const double[:,  ::1] block_A_k_00, block_A_k_01, block_A_k_01T, block_A_k_21, block_A_k_22
    cdef const double[:,  ::1] XAX_kp1_00, XAX_kp1_01, XAX_kp1_01T, XAX_kp1_21, XAX_kp1_22
    cdef double[:, ::1] A_00_workspace1, A_00_workspace2, A_01_workspace1, A_01_workspace2, A_21_workspace1, A_21_workspace2, A_22_workspace1, A_22_workspace2
    cdef double[:, ::1] A_00_workspace1_2, A_00_workspace2_2, A_01_workspace1_2, A_01_workspace2_2, A_21_workspace1_2, A_21_workspace2_2, A_22_workspace1_2, A_22_workspace2_2
    cdef const double[:,  ::1] inv_I
    cdef int r, n, R, total_size  # shape dims
    cdef size_t block_size

    def __init__(self,
                 cnp.ndarray[double, ndim=3] XAX_k_00,
                 cnp.ndarray[double, ndim=3] XAX_k_01,
                 cnp.ndarray[double, ndim=3] XAX_k_21,
                 cnp.ndarray[double, ndim=3] XAX_k_22,
                 cnp.ndarray[double, ndim=4] block_A_k_00,
                 cnp.ndarray[double, ndim=4] block_A_k_01,
                 cnp.ndarray[double, ndim=4] block_A_k_21,
                 cnp.ndarray[double, ndim=4] block_A_k_22,
                 cnp.ndarray[double, ndim=3] XAX_kp1_00,
                 cnp.ndarray[double, ndim=3] XAX_kp1_01,
                 cnp.ndarray[double, ndim=3] XAX_kp1_21,
                 cnp.ndarray[double, ndim=3] XAX_kp1_22,
                 cnp.ndarray[double, ndim=3] inv_I,
                 int r,
                 int n,
                 int R):
        self.XAX_k_00 = np.ascontiguousarray(XAX_k_00.transpose(0, 2, 1).reshape(XAX_k_00.shape[0], -1).T) # rs x l
        self.XAX_k_01 = np.ascontiguousarray(XAX_k_01.transpose(0, 2, 1).reshape(XAX_k_01.shape[0], -1).T)
        self.XAX_k_01T = np.ascontiguousarray(np.transpose(XAX_k_01,  axes=(2, 1, 0)).transpose(0, 2, 1).reshape(XAX_k_01.shape[2], -1).T)
        self.XAX_k_21 = np.ascontiguousarray(XAX_k_21.transpose(0, 2, 1).reshape(XAX_k_21.shape[0], -1).T)
        self.XAX_k_22 = np.ascontiguousarray(XAX_k_22.transpose(0, 2, 1).reshape(XAX_k_22.shape[0], -1).T)

        self.block_A_k_00 = np.ascontiguousarray(block_A_k_00.reshape(block_A_k_00.shape[0] * block_A_k_00.shape[1], block_A_k_00.shape[2] * block_A_k_00.shape[3]).T)
        self.block_A_k_01 = np.ascontiguousarray(block_A_k_01.reshape(block_A_k_01.shape[0] * block_A_k_01.shape[1], block_A_k_01.shape[2] * block_A_k_01.shape[3]).T)
        self.block_A_k_01T = np.ascontiguousarray(np.transpose(block_A_k_01, axes=(0, 2, 1, 3)).reshape(block_A_k_01.shape[0] * block_A_k_01.shape[2], block_A_k_01.shape[1] * block_A_k_01.shape[3]).T)
        self.block_A_k_21 = np.ascontiguousarray(block_A_k_21.reshape(block_A_k_21.shape[0] * block_A_k_21.shape[1], block_A_k_21.shape[2] * block_A_k_21.shape[3]).T)
        self.block_A_k_22 = np.ascontiguousarray(block_A_k_22.reshape(block_A_k_22.shape[0] * block_A_k_22.shape[1], block_A_k_22.shape[2] * block_A_k_22.shape[3]).T)

        self.A_00_workspace1 = np.empty((r * n, R * block_A_k_00.shape[3])) # rn x LS
        self.A_00_workspace1_2 = np.empty((r * R, n * block_A_k_00.shape[3]))
        self.A_00_workspace2 = np.empty((r * R, block_A_k_00.shape[0] * n)) # rL x sm
        self.A_00_workspace2_2 = np.empty((R*n, r*block_A_k_00.shape[0]))

        self.A_01_workspace1 = np.empty((r * n, R * block_A_k_01.shape[3])) # rn x LS
        self.A_01_workspace1_2 = np.empty((r * R, n * block_A_k_01.shape[3]))
        self.A_01_workspace2 = np.empty((r * R, block_A_k_01.shape[0] * n)) # rL x sm
        self.A_01_workspace2_2 = np.empty((R*n, r*block_A_k_01.shape[0]))

        self.A_21_workspace1 = np.empty((r * n, R * block_A_k_21.shape[3])) # rn x LS
        self.A_21_workspace1_2 = np.empty((r * R, n * block_A_k_21.shape[3]))
        self.A_21_workspace2 = np.empty((r * R, block_A_k_21.shape[0] * n)) # rL x sm
        self.A_21_workspace2_2 = np.empty((R*n, r*block_A_k_21.shape[0]))

        self.A_22_workspace1 = np.empty((r * n, R * block_A_k_22.shape[3])) # rn x LS
        self.A_22_workspace1_2 = np.empty((r * R, n * block_A_k_22.shape[3]))
        self.A_22_workspace2 = np.empty((r * R, block_A_k_22.shape[0] * n)) # rL x sm
        self.A_22_workspace2_2 = np.empty((R*n, r*block_A_k_22.shape[0]))

        
        self.XAX_kp1_00 = np.ascontiguousarray(XAX_kp1_00.reshape(-1, R).T) # R x LS
        self.XAX_kp1_01 = np.ascontiguousarray(XAX_kp1_01.reshape(-1, R).T)
        self.XAX_kp1_01T = np.ascontiguousarray(np.transpose(XAX_kp1_01,  axes=(2, 1, 0)).reshape(-1, R).T)
        self.XAX_kp1_21 = np.ascontiguousarray(XAX_kp1_21.reshape(-1, R).T)
        self.XAX_kp1_22 = np.ascontiguousarray(XAX_kp1_22.reshape(-1, R).T)

        self.r = r
        self.n = n
        self.R = R
        self.inv_I = np.ascontiguousarray(inv_I.reshape(self.r*self.n, self.R))

        self.result0 = np.empty((self.R*self.n, self.r), dtype=np.float64)
        self.result1 = np.empty((self.R * self.n, self.r), dtype=np.float64)
        self.temp = np.empty((self.R*self.n, self.r), dtype=np.float64)
        self.x_reshaped_0 = np.empty((self.r*self.n, self.R), dtype=np.float64)
        self.x_reshaped_1 = np.empty((self.r*self.n, self.R), dtype=np.float64)
        self.block_size = self.r * self.n * self.R * sizeof(double)
        self.total_size = 2 * self.r * self.n * self.R
        self.flat_result_arr = np.empty(self.total_size, dtype=np.float64)
        self.flat_result = self.flat_result_arr


    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.nonecheck(False)
    cpdef matvec_into(self, cnp.ndarray[double, ndim=1] x_core, cnp.ndarray[double, ndim=1] out):
        cdef const double[:, :, :] x_reshaped_view = x_core.reshape(2, self.r*self.n, self.R)
        cdef double[:] out_view = out

        with nogil:
            memcpy(&self.x_reshaped_0[0, 0], &x_reshaped_view[0, 0, 0], self.block_size)
            memcpy(&self.x_reshaped_1[0, 0], &x_reshaped_view[1, 0, 0], self.block_size)
            einsum(
                self.XAX_k_00, self.block_A_k_00, self.XAX_kp1_00, 
                self.x_reshaped_0, self.result0, 
                self.A_00_workspace1, self.A_00_workspace1_2, self.A_00_workspace2, self.A_00_workspace2_2,
                self.r, self.n, self.R, 1.0, 0.0
                )
            einsum(
                self.XAX_k_01, self.block_A_k_01, self.XAX_kp1_01, 
                self.x_reshaped_1, self.result0, 
                self.A_01_workspace1, self.A_01_workspace1_2, self.A_01_workspace2, self.A_01_workspace2_2,
                self.r, self.n, self.R, 1.0, 1.0
                )
            einsum(
                self.XAX_k_21, self.block_A_k_21, self.XAX_kp1_21, 
                self.x_reshaped_1, self.result1, 
                self.A_21_workspace1, self.A_21_workspace1_2, self.A_21_workspace2, self.A_21_workspace2_2,
                self.r, self.n, self.R, 1.0, 0.0
                )
            einsum(
                self.XAX_k_01T, self.block_A_k_01T, self.XAX_kp1_01T, 
                self.x_reshaped_0, self.temp, 
                self.A_01_workspace1, self.A_01_workspace1_2, self.A_01_workspace2, self.A_01_workspace2_2,
                self.r, self.n, self.R, 1.0, 0.0
                )
            _transpose_reshape_multiply_inplace(self.temp, self.x_reshaped_0, self.inv_I, self.r, self.n, self.R)
            einsum(
                self.XAX_k_22, self.block_A_k_22, self.XAX_kp1_22, 
                self.x_reshaped_0, self.result1, 
                self.A_22_workspace1, self.A_22_workspace1_2, self.A_22_workspace2, self.A_22_workspace2_2,
                self.r, self.n, self.R, -1.0,  1.0
                )

        pack_results(self.result0, self.result1, out_view, self.R, self.n, self.r)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.nonecheck(False)
    cpdef cnp.ndarray[double, ndim=1] matvec(self, cnp.ndarray[double, ndim=1] x_core):
        self.matvec_into(x_core, self.flat_result_arr)
        return self.flat_result_arr
    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.nonecheck(False)
    cpdef cnp.ndarray[double, ndim=1] rmatvec(self, cnp.ndarray[double, ndim=1] x_core):
        cdef const double[:, :, :] x_reshaped_view = x_core.reshape(2, self.r*self.n, self.R)

        with nogil:
            memcpy(&self.x_reshaped_0[0, 0], &x_reshaped_view[0, 0, 0], self.block_size)
            memcpy(&self.x_reshaped_1[0, 0], &x_reshaped_view[1, 0, 0], self.block_size)
            einsum(
                self.XAX_k_00, self.block_A_k_00, self.XAX_kp1_00, 
                self.x_reshaped_0, self.result0, 
                self.A_00_workspace1, self.A_00_workspace1_2, self.A_00_workspace2, self.A_00_workspace2_2,
                self.r, self.n, self.R, 1.0, 0.0
                )
            einsum(
                self.XAX_k_21, self.block_A_k_21, self.XAX_kp1_21, 
                self.x_reshaped_1, self.result1, 
                self.A_21_workspace1, self.A_21_workspace1_2, self.A_21_workspace2, self.A_21_workspace2_2,
                self.r, self.n, self.R, 1.0, 0.0
                )
            einsum(
                self.XAX_k_01T, self.block_A_k_01T, self.XAX_kp1_01T, 
                self.x_reshaped_0, self.result1, 
                self.A_01_workspace1, self.A_01_workspace1_2, self.A_01_workspace2, self.A_01_workspace2_2,
                self.r, self.n, self.R, 1.0, 1.0
                )
            einsum(
                self.XAX_k_22, self.block_A_k_22, self.XAX_kp1_22, 
                self.x_reshaped_1, self.temp, 
                self.A_22_workspace1, self.A_22_workspace1_2, self.A_22_workspace2, self.A_22_workspace2_2,
                self.r, self.n, self.R, 1.0, 0.0
                )
            _transpose_reshape_multiply_inplace(self.temp, self.x_reshaped_1, self.inv_I, self.r, self.n, self.R)
            einsum(
                self.XAX_k_01, self.block_A_k_01, self.XAX_kp1_01, 
                self.x_reshaped_1, self.result0, 
                self.A_01_workspace1, self.A_01_workspace1_2, self.A_01_workspace2, self.A_01_workspace2_2,
                self.r, self.n, self.R, -1.0,  1.0
                )

        pack_results(self.result0, self.result1, self.flat_result, self.R, self.n, self.r)

        return self.flat_result_arr

cdef class TYMatVecWrapper(BaseMatVec):
    cdef double[:, ::1] result0, result1, temp, x_reshaped_0, x_reshaped_1
    cdef object flat_result_arr
    cdef double[:] flat_result
    cdef const double[:, ::1] XAX_k_00, XAX_k_01, XAX_k_10, XAX_k_21, XAX_k_22
    cdef const double[:, ::1] block_A_k_00, block_A_k_01, block_A_k_10, block_A_k_21, block_A_k_22
    cdef const double[:, ::1] XAX_kp1_00, XAX_kp1_01, XAX_kp1_10, XAX_kp1_21, XAX_kp1_22
    cdef double[:, ::1] A_00_workspace1, A_00_workspace2, A_01_workspace1, A_01_workspace2, A_10_workspace1, A_10_workspace2, A_21_workspace1, A_21_workspace2, A_22_workspace1, A_22_workspace2
    cdef double[:, ::1] A_00_workspace1_2, A_00_workspace2_2, A_01_workspace1_2, A_01_workspace2_2, A_10_workspace1_2, A_10_workspace2_2, A_21_workspace1_2, A_21_workspace2_2, A_22_workspace1_2, A_22_workspace2_2
    cdef const double[:, ::1] inv_I
    cdef int r, n, R, total_size
    cdef size_t block_size

    def __init__(self,
                 cnp.ndarray[double, ndim=3] XAX_k_00,
                 cnp.ndarray[double, ndim=3] XAX_k_01,
                 cnp.ndarray[double, ndim=3] XAX_k_10,
                 cnp.ndarray[double, ndim=3] XAX_k_21,
                 cnp.ndarray[double, ndim=3] XAX_k_22,
                 cnp.ndarray[double, ndim=4] block_A_k_00,
                 cnp.ndarray[double, ndim=4] block_A_k_01,
                 cnp.ndarray[double, ndim=4] block_A_k_10,
                 cnp.ndarray[double, ndim=4] block_A_k_21,
                 cnp.ndarray[double, ndim=4] block_A_k_22,
                 cnp.ndarray[double, ndim=3] XAX_kp1_00,
                 cnp.ndarray[double, ndim=3] XAX_kp1_01,
                 cnp.ndarray[double, ndim=3] XAX_kp1_10,
                 cnp.ndarray[double, ndim=3] XAX_kp1_21,
                 cnp.ndarray[double, ndim=3] XAX_kp1_22,
                 cnp.ndarray[double, ndim=3] inv_I,
                 int r,
                 int n,
                 int R):
        self.XAX_k_00 = np.ascontiguousarray(XAX_k_00.transpose(0, 2, 1).reshape(XAX_k_00.shape[0], -1).T)
        self.XAX_k_01 = np.ascontiguousarray(XAX_k_01.transpose(0, 2, 1).reshape(XAX_k_01.shape[0], -1).T)
        self.XAX_k_10 = np.ascontiguousarray(XAX_k_10.transpose(0, 2, 1).reshape(XAX_k_10.shape[0], -1).T)
        self.XAX_k_21 = np.ascontiguousarray(XAX_k_21.transpose(0, 2, 1).reshape(XAX_k_21.shape[0], -1).T)
        self.XAX_k_22 = np.ascontiguousarray(XAX_k_22.transpose(0, 2, 1).reshape(XAX_k_22.shape[0], -1).T)

        self.block_A_k_00 = np.ascontiguousarray(block_A_k_00.reshape(block_A_k_00.shape[0] * block_A_k_00.shape[1], block_A_k_00.shape[2] * block_A_k_00.shape[3]).T)
        self.block_A_k_01 = np.ascontiguousarray(block_A_k_01.reshape(block_A_k_01.shape[0] * block_A_k_01.shape[1], block_A_k_01.shape[2] * block_A_k_01.shape[3]).T)
        self.block_A_k_10 = np.ascontiguousarray(block_A_k_10.reshape(block_A_k_10.shape[0] * block_A_k_10.shape[1], block_A_k_10.shape[2] * block_A_k_10.shape[3]).T)
        self.block_A_k_21 = np.ascontiguousarray(block_A_k_21.reshape(block_A_k_21.shape[0] * block_A_k_21.shape[1], block_A_k_21.shape[2] * block_A_k_21.shape[3]).T)
        self.block_A_k_22 = np.ascontiguousarray(block_A_k_22.reshape(block_A_k_22.shape[0] * block_A_k_22.shape[1], block_A_k_22.shape[2] * block_A_k_22.shape[3]).T)

        self.A_00_workspace1 = np.empty((r * n, R * block_A_k_00.shape[3]))
        self.A_00_workspace1_2 = np.empty((r * R, n * block_A_k_00.shape[3]))
        self.A_00_workspace2 = np.empty((r * R, block_A_k_00.shape[0] * n))
        self.A_00_workspace2_2 = np.empty((R*n, r*block_A_k_00.shape[0]))

        self.A_01_workspace1 = np.empty((r * n, R * block_A_k_01.shape[3]))
        self.A_01_workspace1_2 = np.empty((r * R, n * block_A_k_01.shape[3]))
        self.A_01_workspace2 = np.empty((r * R, block_A_k_01.shape[0] * n))
        self.A_01_workspace2_2 = np.empty((R*n, r*block_A_k_01.shape[0]))

        self.A_10_workspace1 = np.empty((r * n, R * block_A_k_10.shape[3]))
        self.A_10_workspace1_2 = np.empty((r * R, n * block_A_k_10.shape[3]))
        self.A_10_workspace2 = np.empty((r * R, block_A_k_10.shape[0] * n))
        self.A_10_workspace2_2 = np.empty((R*n, r*block_A_k_10.shape[0]))

        self.A_21_workspace1 = np.empty((r * n, R * block_A_k_21.shape[3]))
        self.A_21_workspace1_2 = np.empty((r * R, n * block_A_k_21.shape[3]))
        self.A_21_workspace2 = np.empty((r * R, block_A_k_21.shape[0] * n))
        self.A_21_workspace2_2 = np.empty((R*n, r*block_A_k_21.shape[0]))

        self.A_22_workspace1 = np.empty((r * n, R * block_A_k_22.shape[3]))
        self.A_22_workspace1_2 = np.empty((r * R, n * block_A_k_22.shape[3]))
        self.A_22_workspace2 = np.empty((r * R, block_A_k_22.shape[0] * n))
        self.A_22_workspace2_2 = np.empty((R*n, r*block_A_k_22.shape[0]))

        self.XAX_kp1_00 = np.ascontiguousarray(XAX_kp1_00.reshape(-1, R).T)
        self.XAX_kp1_01 = np.ascontiguousarray(XAX_kp1_01.reshape(-1, R).T)
        self.XAX_kp1_10 = np.ascontiguousarray(XAX_kp1_10.reshape(-1, R).T)
        self.XAX_kp1_21 = np.ascontiguousarray(XAX_kp1_21.reshape(-1, R).T)
        self.XAX_kp1_22 = np.ascontiguousarray(XAX_kp1_22.reshape(-1, R).T)

        self.r = r
        self.n = n
        self.R = R
        self.inv_I = np.ascontiguousarray(inv_I.reshape(self.r*self.n, self.R))

        self.result0 = np.empty((self.R*self.n, self.r), dtype=np.float64)
        self.result1 = np.empty((self.R*self.n, self.r), dtype=np.float64)
        self.temp = np.empty((self.R*self.n, self.r), dtype=np.float64)
        self.x_reshaped_0 = np.empty((self.r*self.n, self.R), dtype=np.float64)
        self.x_reshaped_1 = np.empty((self.r*self.n, self.R), dtype=np.float64)
        self.block_size = self.r * self.n * self.R * sizeof(double)
        self.total_size = 2 * self.r * self.n * self.R
        self.flat_result_arr = np.empty(self.total_size, dtype=np.float64)
        self.flat_result = self.flat_result_arr

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.nonecheck(False)
    cpdef matvec_into(self, cnp.ndarray[double, ndim=1] x_core, cnp.ndarray[double, ndim=1] out):
        cdef const double[:, :, :] x_reshaped_view = x_core.reshape(2, self.r*self.n, self.R)
        cdef double[:] out_view = out

        with nogil:
            memcpy(&self.x_reshaped_0[0, 0], &x_reshaped_view[0, 0, 0], self.block_size)
            memcpy(&self.x_reshaped_1[0, 0], &x_reshaped_view[1, 0, 0], self.block_size)

            einsum(self.XAX_k_00, self.block_A_k_00, self.XAX_kp1_00, self.x_reshaped_0, self.result0, self.A_00_workspace1, self.A_00_workspace1_2, self.A_00_workspace2, self.A_00_workspace2_2, self.r, self.n, self.R, 1.0, 0.0)
            einsum(self.XAX_k_01, self.block_A_k_01, self.XAX_kp1_01, self.x_reshaped_1, self.result0, self.A_01_workspace1, self.A_01_workspace1_2, self.A_01_workspace2, self.A_01_workspace2_2, self.r, self.n, self.R, 1.0, 1.0)

            einsum(self.XAX_k_21, self.block_A_k_21, self.XAX_kp1_21, self.x_reshaped_1, self.result1, self.A_21_workspace1, self.A_21_workspace1_2, self.A_21_workspace2, self.A_21_workspace2_2, self.r, self.n, self.R, 1.0, 0.0)
            einsum(self.XAX_k_10, self.block_A_k_10, self.XAX_kp1_10, self.x_reshaped_0, self.temp, self.A_10_workspace1, self.A_10_workspace1_2, self.A_10_workspace2, self.A_10_workspace2_2, self.r, self.n, self.R, 1.0, 0.0)
            _transpose_reshape_multiply_inplace(self.temp, self.x_reshaped_0, self.inv_I, self.r, self.n, self.R)
            einsum(self.XAX_k_22, self.block_A_k_22, self.XAX_kp1_22, self.x_reshaped_0, self.result1, self.A_22_workspace1, self.A_22_workspace1_2, self.A_22_workspace2, self.A_22_workspace2_2, self.r, self.n, self.R, -1.0, 1.0)

        pack_results(self.result0, self.result1, out_view, self.R, self.n, self.r)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.nonecheck(False)
    cpdef cnp.ndarray[double, ndim=1] matvec(self, cnp.ndarray[double, ndim=1] x_core):
        self.matvec_into(x_core, self.flat_result_arr)
        return self.flat_result_arr

cdef class IneqMatVecWrapper(BaseMatVec):
    cdef double[:,  ::1] result0, result1, result2, temp, x_reshaped_0, x_reshaped_1, x_reshaped_2
    cdef object flat_result_arr
    cdef double[:] flat_result
    cdef double[:,  ::1] XAX_k_00, XAX_k_01, XAX_k_01T, XAX_k_21, XAX_k_22, XAX_k_31, XAX_k_33
    cdef double[:,  ::1] block_A_k_00, block_A_k_01, block_A_k_01T, block_A_k_21, block_A_k_22, block_A_k_31, block_A_k_33
    cdef double[:,  ::1] XAX_kp1_00, XAX_kp1_01, XAX_kp1_01T, XAX_kp1_21, XAX_kp1_22, XAX_kp1_31, XAX_kp1_33
    cdef double[:, ::1] inv_I
    cdef double[:, ::1] A_00_workspace1, A_00_workspace2, A_01_workspace1, A_01_workspace2, A_21_workspace1, A_21_workspace2, A_22_workspace1, A_22_workspace2, A_31_workspace1, A_31_workspace2, A_33_workspace1, A_33_workspace2
    cdef double[:, ::1] A_00_workspace1_2, A_00_workspace2_2, A_01_workspace1_2, A_01_workspace2_2, A_21_workspace1_2, A_21_workspace2_2, A_22_workspace1_2, A_22_workspace2_2, A_31_workspace1_2, A_31_workspace2_2, A_33_workspace1_2, A_33_workspace2_2
    cdef int r, n, R, total_size
    cdef size_t block_size

    def __init__(self,
                 cnp.ndarray[double, ndim=3] XAX_k_00,
                 cnp.ndarray[double, ndim=3] XAX_k_01,
                 cnp.ndarray[double, ndim=3] XAX_k_21,
                 cnp.ndarray[double, ndim=3] XAX_k_22,
                 cnp.ndarray[double, ndim=3] XAX_k_31,
                 cnp.ndarray[double, ndim=3] XAX_k_33,
                 cnp.ndarray[double, ndim=4] block_A_k_00,
                 cnp.ndarray[double, ndim=4] block_A_k_01,
                 cnp.ndarray[double, ndim=4] block_A_k_21,
                 cnp.ndarray[double, ndim=4] block_A_k_22,
                 cnp.ndarray[double, ndim=4] block_A_k_31,
                 cnp.ndarray[double, ndim=4] block_A_k_33,
                 cnp.ndarray[double, ndim=3] XAX_kp1_00,
                 cnp.ndarray[double, ndim=3] XAX_kp1_01,
                 cnp.ndarray[double, ndim=3] XAX_kp1_21,
                 cnp.ndarray[double, ndim=3] XAX_kp1_22,
                 cnp.ndarray[double, ndim=3] XAX_kp1_31,
                 cnp.ndarray[double, ndim=3] XAX_kp1_33,
                 cnp.ndarray[double, ndim=3] inv_I,
                 int r,
                 int n,
                 int R):

        self.XAX_k_00 = np.ascontiguousarray(XAX_k_00.transpose(0, 2, 1).reshape(XAX_k_00.shape[0], -1).T)
        self.XAX_k_01 = np.ascontiguousarray(XAX_k_01.transpose(0, 2, 1).reshape(XAX_k_01.shape[0], -1).T)
        self.XAX_k_01T = np.ascontiguousarray(np.transpose(XAX_k_01,  axes=(2, 1, 0)).transpose(0, 2, 1).reshape(XAX_k_01.shape[2], -1).T)
        self.XAX_k_21 = np.ascontiguousarray(XAX_k_21.transpose(0, 2, 1).reshape(XAX_k_21.shape[0], -1).T)
        self.XAX_k_22 = np.ascontiguousarray(XAX_k_22.transpose(0, 2, 1).reshape(XAX_k_22.shape[0], -1).T)
        self.XAX_k_31 = np.ascontiguousarray(XAX_k_31.transpose(0, 2, 1).reshape(XAX_k_31.shape[0], -1).T)
        self.XAX_k_33 = np.ascontiguousarray(XAX_k_33.transpose(0, 2, 1).reshape(XAX_k_33.shape[0], -1).T)

        self.block_A_k_00 = np.ascontiguousarray(block_A_k_00.reshape(block_A_k_00.shape[0] * block_A_k_00.shape[1], block_A_k_00.shape[2] * block_A_k_00.shape[3]).T)
        self.block_A_k_01 = np.ascontiguousarray(block_A_k_01.reshape(block_A_k_01.shape[0] * block_A_k_01.shape[1], block_A_k_01.shape[2] * block_A_k_01.shape[3]).T)
        self.block_A_k_01T = np.ascontiguousarray(np.transpose(block_A_k_01, axes=(0, 2, 1, 3)).reshape(block_A_k_01.shape[0] * block_A_k_01.shape[2], block_A_k_01.shape[1] * block_A_k_01.shape[3]).T)
        self.block_A_k_21 = np.ascontiguousarray(block_A_k_21.reshape(block_A_k_21.shape[0] * block_A_k_21.shape[1], block_A_k_21.shape[2] * block_A_k_21.shape[3]).T)
        self.block_A_k_22 = np.ascontiguousarray(block_A_k_22.reshape(block_A_k_22.shape[0] * block_A_k_22.shape[1], block_A_k_22.shape[2] * block_A_k_22.shape[3]).T)
        self.block_A_k_31 = np.ascontiguousarray(block_A_k_31.reshape(block_A_k_31.shape[0] * block_A_k_31.shape[1], block_A_k_31.shape[2] * block_A_k_31.shape[3]).T)
        self.block_A_k_33 = np.ascontiguousarray(block_A_k_33.reshape(block_A_k_33.shape[0] * block_A_k_33.shape[1], block_A_k_33.shape[2] * block_A_k_33.shape[3]).T)

        self.A_00_workspace1 = np.empty((r * n, R * block_A_k_00.shape[3])) # rn x LS
        self.A_00_workspace1_2 = np.empty((r * R, n * block_A_k_00.shape[3]))
        self.A_00_workspace2 = np.empty((r * R, block_A_k_00.shape[0] * n)) # rL x sm
        self.A_00_workspace2_2 = np.empty((R*n, r*block_A_k_00.shape[0]))

        self.A_01_workspace1 = np.empty((r * n, R * block_A_k_01.shape[3])) # rn x LS
        self.A_01_workspace1_2 = np.empty((r * R, n * block_A_k_01.shape[3]))
        self.A_01_workspace2 = np.empty((r * R, block_A_k_01.shape[0] * n)) # rL x sm
        self.A_01_workspace2_2 = np.empty((R*n, r*block_A_k_01.shape[0]))

        self.A_21_workspace1 = np.empty((r * n, R * block_A_k_21.shape[3])) # rn x LS
        self.A_21_workspace1_2 = np.empty((r * R, n * block_A_k_21.shape[3]))
        self.A_21_workspace2 = np.empty((r * R, block_A_k_21.shape[0] * n)) # rL x sm
        self.A_21_workspace2_2 = np.empty((R*n, r*block_A_k_21.shape[0]))

        self.A_22_workspace1 = np.empty((r * n, R * block_A_k_22.shape[3])) # rn x LS
        self.A_22_workspace1_2 = np.empty((r * R, n * block_A_k_22.shape[3]))
        self.A_22_workspace2 = np.empty((r * R, block_A_k_22.shape[0] * n)) # rL x sm
        self.A_22_workspace2_2 = np.empty((R*n, r*block_A_k_22.shape[0]))

        self.A_31_workspace1 = np.empty((r * n, R * block_A_k_31.shape[3])) # rn x LS
        self.A_31_workspace1_2 = np.empty((r * R, n * block_A_k_31.shape[3]))
        self.A_31_workspace2 = np.empty((r * R, block_A_k_31.shape[0] * n)) # rL x sm
        self.A_31_workspace2_2 = np.empty((R*n, r*block_A_k_31.shape[0]))

        self.A_33_workspace1 = np.empty((r * n, R * block_A_k_33.shape[3])) # rn x LS
        self.A_33_workspace1_2 = np.empty((r * R, n * block_A_k_33.shape[3]))
        self.A_33_workspace2 = np.empty((r * R, block_A_k_33.shape[0] * n)) # rL x sm
        self.A_33_workspace2_2 = np.empty((R*n, r*block_A_k_33.shape[0]))
        
        self.XAX_kp1_00 = np.ascontiguousarray(XAX_kp1_00.reshape(-1, R).T)
        self.XAX_kp1_01 = np.ascontiguousarray(XAX_kp1_01.reshape(-1, R).T)
        self.XAX_kp1_01T = np.ascontiguousarray(np.transpose(XAX_kp1_01,  axes=(2, 1, 0)).reshape(-1, R).T)
        self.XAX_kp1_21 = np.ascontiguousarray(XAX_kp1_21.reshape(-1, R).T)
        self.XAX_kp1_22 = np.ascontiguousarray(XAX_kp1_22.reshape(-1, R).T)
        self.XAX_kp1_31 = np.ascontiguousarray(XAX_kp1_31.reshape(-1, R).T)
        self.XAX_kp1_33 = np.ascontiguousarray(XAX_kp1_33.reshape(-1, R).T)

        self.r = r
        self.n = n
        self.R = R
        self.inv_I = np.ascontiguousarray(inv_I.reshape(self.r*self.n, self.R))

        self.result0 = np.empty((self.R * self.n, self.r), dtype=np.float64)
        self.result1 = np.empty((self.R * self.n, self.r), dtype=np.float64)
        self.result2 = np.empty((self.R * self.n, self.r), dtype=np.float64)
        self.temp = np.empty((self.R*self.n, self.r), dtype=np.float64)
        self.x_reshaped_0 = np.empty((self.r*self.n, self.R), dtype=np.float64)
        self.x_reshaped_1 = np.empty((self.r*self.n, self.R), dtype=np.float64)
        self.x_reshaped_2 = np.empty((self.r*self.n, self.R), dtype=np.float64)
        self.block_size = self.r * self.n * self.R * sizeof(double)
        self.total_size = 3 * self.r * self.n * self.R
        self.flat_result_arr = np.empty(self.total_size, dtype=np.float64)
        self.flat_result = self.flat_result_arr

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.nonecheck(False)
    cpdef matvec_into(self, cnp.ndarray[double, ndim=1] x_core, cnp.ndarray[double, ndim=1] out):
        cdef const double[:, :, :] x_reshaped_view = x_core.reshape(3, self.r*self.n, self.R)
        cdef double[:] out_view = out

        with nogil:
            memcpy(&self.x_reshaped_0[0, 0], &x_reshaped_view[0, 0, 0], self.block_size)
            memcpy(&self.x_reshaped_1[0, 0], &x_reshaped_view[1, 0, 0], self.block_size)
            memcpy(&self.x_reshaped_2[0, 0], &x_reshaped_view[2, 0, 0], self.block_size)

            einsum(self.XAX_k_00, self.block_A_k_00, self.XAX_kp1_00, self.x_reshaped_0, self.result0, self.A_00_workspace1, self.A_00_workspace1_2, self.A_00_workspace2, self.A_00_workspace2_2, self.r, self.n, self.R, 1.0, 0.0)
            einsum(self.XAX_k_01, self.block_A_k_01, self.XAX_kp1_01, self.x_reshaped_1, self.result0, self.A_01_workspace1, self.A_01_workspace1_2, self.A_01_workspace2, self.A_01_workspace2_2, self.r, self.n, self.R, 1.0, 1.0)
            
            einsum(self.XAX_k_21, self.block_A_k_21, self.XAX_kp1_21, self.x_reshaped_1, self.result1, self.A_21_workspace1, self.A_21_workspace1_2, self.A_21_workspace2, self.A_21_workspace2_2, self.r, self.n, self.R, 1.0, 0.0)
            einsum(self.XAX_k_01T, self.block_A_k_01T, self.XAX_kp1_01T, self.x_reshaped_0, self.temp, self.A_01_workspace1, self.A_01_workspace1_2, self.A_01_workspace2, self.A_01_workspace2_2, self.r, self.n, self.R, 1.0, 0.0)
            _transpose_reshape_multiply_inplace(self.temp, self.x_reshaped_0, self.inv_I, self.r, self.n, self.R)
            cy_maxpy(self.x_reshaped_2, self.x_reshaped_0)
            einsum(self.XAX_k_22, self.block_A_k_22, self.XAX_kp1_22, self.x_reshaped_0, self.result1, self.A_22_workspace1, self.A_22_workspace1_2, self.A_22_workspace2, self.A_22_workspace2_2, self.r, self.n, self.R, -1.0, 1.0)
            
            einsum(self.XAX_k_31, self.block_A_k_31, self.XAX_kp1_31, self.x_reshaped_1, self.result2, self.A_31_workspace1, self.A_31_workspace1_2, self.A_31_workspace2, self.A_31_workspace2_2, self.r, self.n, self.R, 1.0, 0.0)
            einsum(self.XAX_k_33, self.block_A_k_33, self.XAX_kp1_33, self.x_reshaped_2, self.result2, self.A_33_workspace1, self.A_33_workspace1_2, self.A_33_workspace2, self.A_33_workspace2_2, self.r, self.n, self.R, 1.0, 1.0)
        pack_results3(self.result0, self.result1, self.result2, out_view, self.R, self.n, self.r)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.nonecheck(False)
    cpdef cnp.ndarray[double, ndim=1] matvec(self, cnp.ndarray[double, ndim=1] x_core):
        self.matvec_into(x_core, self.flat_result_arr)
        return self.flat_result_arr
