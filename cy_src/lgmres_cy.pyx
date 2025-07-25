# lgmres_cy.pyx
# cython: language_level=3
# cython: cdivision=True
# cython: optimize.use_switch=True
# distutils: language = c++

cdef extern from "numpy/arrayobject.h":
    # Define NPY_NO_DEPRECATED_API for compatibility with numpy
    ctypedef void npy_no_deprecated_api


cdef extern from "cblas.h" nogil:
    # --- Enums for Layout and Transpose ---
    ctypedef enum CBLAS_LAYOUT:
        CblasRowMajor=101
        CblasColMajor=102

    ctypedef enum CBLAS_TRANSPOSE:
        CblasNoTrans=111
        CblasTrans=112
        CblasConjTrans=113

    void cblas_dgemm(const CBLAS_LAYOUT Layout, const CBLAS_TRANSPOSE TransA,
                     const CBLAS_TRANSPOSE TransB, const int M, const int N,
                     const int K, const double alpha, const double* A,
                     const int lda, const double* B, const int ldb,
                     const double beta, double* C, const int ldc)

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
    cdef int lda = A.shape[1]
    cdef int ldb = B.shape[1]
    cdef int ldc = C.shape[1]
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, M, ldb, lda,
                alpha, &A[0, 0], lda, &B[0, 0], ldb,
                beta, &C[0, 0], ldc)


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
    cdef int i, j, k, batch
    cdef int idx

    for i in range(r):
        for j in range(n):
            for k in range(R):
                idx = (i * n + j) * R + k  # flatten index
                flat_result[idx] = result0[k * n + j, i]

    for i in range(r):
        for j in range(n):
            for k in range(R):
                idx = ((r + i) * n + j) * R + k  # flatten index
                flat_result[idx] = result1[k * n + j, i]


cdef class BaseMatVec:

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.nonecheck(False)
    cpdef cnp.ndarray[double, ndim=1] matvec(self, cnp.ndarray[double, ndim=1] x_core):
        raise NotImplementedError("BaseMatVec.matvec must be implemented by subclass")

cdef class MatVecWrapper(BaseMatVec):
    cdef double[:,  ::1] result0, result1, temp, x_reshaped_0, x_reshaped_1
    cdef const double[:,  ::1] XAX_k_00, XAX_k_01, XAX_k_01T, XAX_k_21, XAX_k_22
    cdef const double[:,  ::1] block_A_k_00, block_A_k_01, block_A_k_01T, block_A_k_21, block_A_k_22
    cdef const double[:,  ::1] XAX_kp1_00, XAX_kp1_01, XAX_kp1_01T, XAX_kp1_21, XAX_kp1_22
    cdef double[:, ::1] A_00_workspace1, A_00_workspace2, A_01_workspace1, A_01_workspace2,A_01T_workspace1, A_01T_workspace2, A_21_workspace1, A_21_workspace2, A_22_workspace1, A_22_workspace2
    cdef double[:, ::1] A_00_workspace1_2, A_00_workspace2_2, A_01_workspace1_2, A_01_workspace2_2,A_01T_workspace1_2, A_01T_workspace2_2, A_21_workspace1_2, A_21_workspace2_2, A_22_workspace1_2, A_22_workspace2_2
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

        self.A_01T_workspace1 = np.empty((r * n, R * block_A_k_01.shape[3])) # rn x LS
        self.A_01T_workspace1_2 = np.empty((r * R, n * block_A_k_01.shape[3]))
        self.A_01T_workspace2 = np.empty((r * R, block_A_k_01.shape[0] * n)) # rL x sm
        self.A_01T_workspace2_2 = np.empty((R*n, r*block_A_k_01.shape[0]))

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


    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.nonecheck(False)
    cpdef cnp.ndarray[double, ndim=1] matvec(self, cnp.ndarray[double, ndim=1] x_core):
        cdef double[:, :, :] x_reshaped_view = x_core.view().reshape(2, self.r*self.n, self.R)

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
                self.A_01T_workspace1, self.A_01T_workspace1_2, self.A_01T_workspace2, self.A_01T_workspace2_2,
                self.r, self.n, self.R, 1.0, 0.0
                )
            _transpose_reshape_multiply_inplace(self.temp, self.x_reshaped_0, self.inv_I, self.r, self.n, self.R)
            einsum(
                self.XAX_k_22, self.block_A_k_22, self.XAX_kp1_22, 
                self.x_reshaped_0, self.result1, 
                self.A_22_workspace1, self.A_22_workspace1_2, self.A_22_workspace2, self.A_22_workspace2_2,
                self.r, self.n, self.R, -1.0,  1.0
                )

        cdef cnp.ndarray[double, ndim=1] flat_result = np.empty(self.total_size, dtype=np.float64)
        pack_results(self.result0, self.result1, flat_result, self.R, self.n, self.r)
                
        return flat_result

cdef class IneqMatVecWrapper(BaseMatVec):
    cdef double[:,  ::1] result0, result1, result2, temp, x_reshaped_0, x_reshaped_1, x_reshaped_2
    cdef double[:,  ::1] XAX_k_00, XAX_k_01, XAX_k_01T, XAX_k_21, XAX_k_22, XAX_k_31, XAX_k_33
    cdef double[:,  ::1] block_A_k_00, block_A_k_01, block_A_k_01T, block_A_k_21, block_A_k_22, block_A_k_31, block_A_k_33
    cdef double[:,  ::1] XAX_kp1_00, XAX_kp1_01, XAX_kp1_01T, XAX_kp1_21, XAX_kp1_22, XAX_kp1_31, XAX_kp1_33
    cdef double[:, ::1] inv_I
    cdef double[:, ::1] A_00_workspace1, A_00_workspace2, A_01_workspace1, A_01_workspace2,A_01T_workspace1, A_01T_workspace2, A_21_workspace1, A_21_workspace2, A_22_workspace1, A_22_workspace2, A_31_workspace1, A_31_workspace2, A_33_workspace1, A_33_workspace2
    cdef double[:, ::1] A_00_workspace1_2, A_00_workspace2_2, A_01_workspace1_2, A_01_workspace2_2,A_01T_workspace1_2, A_01T_workspace2_2, A_21_workspace1_2, A_21_workspace2_2, A_22_workspace1_2, A_22_workspace2_2, A_31_workspace1_2, A_31_workspace2_2, A_33_workspace1_2, A_33_workspace2_2
    cdef int r, n, R
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

        self.A_01T_workspace1 = np.empty((r * n, R * block_A_k_01.shape[3])) # rn x LS
        self.A_01T_workspace1_2 = np.empty((r * R, n * block_A_k_01.shape[3]))
        self.A_01T_workspace2 = np.empty((r * R, block_A_k_01.shape[0] * n)) # rL x sm
        self.A_01T_workspace2_2 = np.empty((R*n, r*block_A_k_01.shape[0]))

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

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.nonecheck(False)
    cpdef cnp.ndarray[double, ndim=1] matvec(self, cnp.ndarray[double, ndim=1] x_core):
        cdef cnp.ndarray[double, ndim=4] result = np.empty((3, self.r, self.n, self.R), dtype=np.float64)
        cdef double[:, :, :] x_reshaped_view = x_core.view().reshape(3, self.r*self.n, self.R)

        with nogil:
            memcpy(&self.x_reshaped_0[0, 0], &x_reshaped_view[0, 0, 0], self.block_size)
            memcpy(&self.x_reshaped_1[0, 0], &x_reshaped_view[1, 0, 0], self.block_size)
            memcpy(&self.x_reshaped_2[0, 0], &x_reshaped_view[2, 0, 0], self.block_size)

            einsum(self.XAX_k_00, self.block_A_k_00, self.XAX_kp1_00, self.x_reshaped_0, self.result0, self.A_00_workspace1, self.A_00_workspace1_2, self.A_00_workspace2, self.A_00_workspace2_2, self.r, self.n, self.R, 1.0, 0.0)
            einsum(self.XAX_k_01, self.block_A_k_01, self.XAX_kp1_01, self.x_reshaped_1, self.result0, self.A_01_workspace1, self.A_01_workspace1_2, self.A_01_workspace2, self.A_01_workspace2_2, self.r, self.n, self.R, 1.0, 1.0)
            
            einsum(self.XAX_k_21, self.block_A_k_21, self.XAX_kp1_21, self.x_reshaped_1, self.result1, self.A_21_workspace1, self.A_21_workspace1_2, self.A_21_workspace2, self.A_21_workspace2_2, self.r, self.n, self.R, 1.0, 0.0)
            einsum(self.XAX_k_01T, self.block_A_k_01T, self.XAX_kp1_01T, self.x_reshaped_0, self.temp, self.A_01T_workspace1, self.A_01T_workspace1_2, self.A_01T_workspace2, self.A_01T_workspace2_2, self.r, self.n, self.R, 1.0, 0.0)
            _transpose_reshape_multiply_inplace(self.temp, self.x_reshaped_0, self.inv_I, self.r, self.n, self.R)
            cy_maxpy(self.x_reshaped_2, self.x_reshaped_0)
            einsum(self.XAX_k_22, self.block_A_k_22, self.XAX_kp1_22, self.x_reshaped_0, self.result1, self.A_22_workspace1, self.A_22_workspace1_2, self.A_22_workspace2, self.A_22_workspace2_2, self.r, self.n, self.R, -1.0, 1.0)
            
            einsum(self.XAX_k_31, self.block_A_k_31, self.XAX_kp1_31, self.x_reshaped_1, self.result2, self.A_31_workspace1, self.A_31_workspace1_2, self.A_31_workspace2, self.A_31_workspace2_2, self.r, self.n, self.R, 1.0, 0.0)
            einsum(self.XAX_k_33, self.block_A_k_33, self.XAX_kp1_33, self.x_reshaped_2, self.result2, self.A_33_workspace1, self.A_33_workspace1_2, self.A_33_workspace2, self.A_33_workspace2_2, self.r, self.n, self.R, 1.0, 1.0)
        
        result[0] = np.asarray(self.result0).reshape(self.R, self.n, self.r).transpose(2, 1, 0)
        result[1] = np.asarray(self.result1).reshape(self.R, self.n, self.r).transpose(2, 1, 0)
        result[2] = np.asarray(self.result2).reshape(self.R, self.n, self.r).transpose(2, 1, 0)
        return result.ravel()