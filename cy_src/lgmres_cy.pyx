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
cimport scipy.linalg.cython_lapack as lapack
from libc.string cimport memcpy
from libc.math cimport fabs, sqrt
from scipy.linalg.cython_blas cimport dcopy

cnp.import_array() # Initialize NumPy C-API

cdef:
    int inc = 1  # typical unit stride
    double global_alpha = 1.0


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.inline
cdef bint _prefer_block_c(
        double val,
        double warm_norm,
        double best_val,
        double best_warm_norm,
        bint has_best,
        double eps
) noexcept nogil:
    cdef double scale = fabs(val)
    cdef double best_abs = fabs(best_val)
    cdef double tol
    if not has_best:
        return True
    if best_abs > scale:
        scale = best_abs
    if scale < 1.0:
        scale = 1.0
    tol = 10.0 * eps
    if tol < 1e-12:
        tol = 1e-12
    tol *= scale
    return val < best_val - tol or (fabs(val - best_val) <= tol and warm_norm > best_warm_norm)


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
cdef int _sym_smallest_eigh(
        double[:, ::1] block,
        double[::1] eigvals,
        double[::1] work,
        double* value
) noexcept nogil:
    cdef int n = block.shape[0]
    cdef int lda = n
    cdef int lwork = work.shape[0]
    cdef int info = 0
    cdef int i, j
    cdef double v
    cdef char jobz = 86
    cdef char uplo = 85
    if n == 1:
        value[0] = block[0, 0]
        block[0, 0] = 1.0
        return 0
    for i in range(n):
        for j in range(i + 1, n):
            v = 0.5 * (block[i, j] + block[j, i])
            block[i, j] = v
            block[j, i] = v
    lapack.dsyev(&jobz, &uplo, &n, &block[0, 0], &lda, &eigvals[0], &work[0], &lwork, &info)
    value[0] = eigvals[0]
    return info

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
cpdef object dense_scaled_min_eig(
        cnp.ndarray[double, ndim=2] A_in,
        cnp.ndarray[double, ndim=2] D_in,
        double scale
):
    cdef int n = A_in.shape[0]
    cdef int work_size = 3 * n - 1
    cdef cnp.ndarray[double, ndim=2] A_arr
    cdef cnp.ndarray[double, ndim=2] D_arr
    cdef cnp.ndarray[double, ndim=2] block_arr
    cdef cnp.ndarray[double, ndim=1] eigvals_arr
    cdef cnp.ndarray[double, ndim=1] work_arr
    cdef cnp.ndarray[double, ndim=1] sol_arr
    cdef cnp.ndarray[double, ndim=1] y_arr
    cdef const double[:, ::1] A, D
    cdef double[:, ::1] block
    cdef double[::1] eigvals, work, sol, y
    cdef double eig_val = 0.0
    cdef double residual = 0.0
    cdef double y_norm = 0.0
    cdef double vec_norm = 0.0
    cdef double entry
    cdef int i, j, info
    if n <= 0 or A_in.shape[1] != n or D_in.shape[0] != n or D_in.shape[1] != n:
        return None
    if work_size < 1:
        work_size = 1
    A_arr = np.ascontiguousarray(A_in, dtype=np.float64)
    D_arr = np.ascontiguousarray(D_in, dtype=np.float64)
    block_arr = np.empty((n, n), dtype=np.float64)
    eigvals_arr = np.empty(n, dtype=np.float64)
    work_arr = np.empty(work_size, dtype=np.float64)
    sol_arr = np.empty(n, dtype=np.float64)
    y_arr = np.empty(n, dtype=np.float64)
    A = A_arr
    D = D_arr
    block = block_arr
    eigvals = eigvals_arr
    work = work_arr
    sol = sol_arr
    y = y_arr
    with nogil:
        for i in range(n):
            for j in range(n):
                block[i, j] = scale * A[i, j] + D[i, j]
        info = _sym_smallest_eigh(block, eigvals, work, &eig_val)
        if info == 0:
            for i in range(n):
                sol[i] = block[0, i]
            cy_dgemv_row(A, sol, y, scale, 0.0)
            cy_dgemv_row(D, sol, y, 1.0, 1.0)
            for i in range(n):
                entry = y[i] - eig_val * sol[i]
                residual += entry * entry
                y_norm += y[i] * y[i]
                vec_norm += sol[i] * sol[i]
    if info != 0:
        return None
    residual = sqrt(residual)
    y_norm = sqrt(y_norm)
    vec_norm = sqrt(vec_norm)
    y_norm = max(y_norm, fabs(eig_val) * vec_norm, 1.0)
    return eig_val, sol_arr.reshape(n, 1), residual / y_norm


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
cpdef tuple dense_schur_3block(
        cnp.ndarray[double, ndim=2] A00_in,
        cnp.ndarray[double, ndim=2] A01_in,
        cnp.ndarray[double, ndim=2] A10_in,
        cnp.ndarray[double, ndim=2] base_in,
        cnp.ndarray[double, ndim=2] rhs_p_in,
        cnp.ndarray[double, ndim=2] rhs_x_in
):
    cdef int n = A00_in.shape[0]
    cdef int nrhs = rhs_p_in.shape[1]
    cdef cnp.ndarray[double, ndim=2] A00_arr = np.ascontiguousarray(A00_in, dtype=np.float64)
    cdef cnp.ndarray[double, ndim=2] A01_arr = np.ascontiguousarray(A01_in, dtype=np.float64)
    cdef cnp.ndarray[double, ndim=2] A10_arr = np.ascontiguousarray(A10_in, dtype=np.float64)
    cdef cnp.ndarray[double, ndim=2] base_arr = np.ascontiguousarray(base_in, dtype=np.float64)
    cdef cnp.ndarray[double, ndim=2] rhs_p_arr = np.ascontiguousarray(rhs_p_in, dtype=np.float64)
    cdef cnp.ndarray[double, ndim=2] rhs_x_arr = np.ascontiguousarray(rhs_x_in, dtype=np.float64)
    cdef cnp.ndarray[double, ndim=2] tmp_arr = np.empty((n, n), dtype=np.float64)
    cdef cnp.ndarray[double, ndim=2] lhs_arr = np.empty((n, n), dtype=np.float64)
    cdef cnp.ndarray[double, ndim=2] b_arr = np.empty((n, nrhs), dtype=np.float64)
    cdef const double[:, ::1] A00 = A00_arr
    cdef const double[:, ::1] A01 = A01_arr
    cdef const double[:, ::1] A10 = A10_arr
    cdef const double[:, ::1] base = base_arr
    cdef const double[:, ::1] rhs_p = rhs_p_arr
    cdef const double[:, ::1] rhs_x = rhs_x_arr
    cdef double[:, ::1] tmp = tmp_arr
    cdef double[:, ::1] lhs = lhs_arr
    cdef double[:, ::1] b = b_arr
    cdef int i, j
    with nogil:
        cy_dgemm(A01, base, tmp)
        cy_dgemm(tmp, A10, lhs)
        for i in range(n):
            for j in range(n):
                lhs[i, j] += A00[i, j]
        cy_dgemm(A01, rhs_x, b)
        for i in range(n):
            for j in range(nrhs):
                b[i, j] = rhs_p[i, j] - b[i, j]
    return lhs_arr, b_arr


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
cpdef tuple dense_schur_4block(
        cnp.ndarray[double, ndim=2] A00_in,
        cnp.ndarray[double, ndim=2] D33_in,
        cnp.ndarray[double, ndim=2] Aeq_in,
        cnp.ndarray[double, ndim=2] Top_in,
        cnp.ndarray[double, ndim=2] Linv_rhs_in,
        cnp.ndarray[double, ndim=2] Linv_X_in,
        cnp.ndarray[double, ndim=1] inv_I_in,
        cnp.ndarray[double, ndim=2] rhs_p_in,
        cnp.ndarray[double, ndim=2] rhs_d_in,
        cnp.ndarray[double, ndim=2] rhs_t_in
):
    cdef int n = A00_in.shape[0]
    cdef cnp.ndarray[double, ndim=2] A00_arr = np.ascontiguousarray(A00_in, dtype=np.float64)
    cdef cnp.ndarray[double, ndim=2] D33_arr = np.ascontiguousarray(D33_in, dtype=np.float64)
    cdef cnp.ndarray[double, ndim=2] Aeq_arr = np.ascontiguousarray(Aeq_in, dtype=np.float64)
    cdef cnp.ndarray[double, ndim=2] Top_arr = np.ascontiguousarray(Top_in, dtype=np.float64)
    cdef cnp.ndarray[double, ndim=2] Linv_rhs_arr = np.ascontiguousarray(Linv_rhs_in, dtype=np.float64)
    cdef cnp.ndarray[double, ndim=2] Linv_X_arr = np.ascontiguousarray(Linv_X_in, dtype=np.float64)
    cdef cnp.ndarray[double, ndim=1] inv_arr = np.ascontiguousarray(inv_I_in, dtype=np.float64)
    cdef cnp.ndarray[double, ndim=2] rhs_p_arr = np.ascontiguousarray(rhs_p_in, dtype=np.float64)
    cdef cnp.ndarray[double, ndim=2] rhs_d_arr = np.ascontiguousarray(rhs_d_in, dtype=np.float64)
    cdef cnp.ndarray[double, ndim=2] rhs_t_arr = np.ascontiguousarray(rhs_t_in, dtype=np.float64)
    cdef cnp.ndarray[double, ndim=2] Aeq_T_arr = np.ascontiguousarray(Aeq_arr.T)
    cdef cnp.ndarray[double, ndim=2] scaled_arr = np.empty((n, n), dtype=np.float64)
    cdef cnp.ndarray[double, ndim=2] tmp_mat_arr = np.empty((n, n), dtype=np.float64)
    cdef cnp.ndarray[double, ndim=2] tmp_rhs_arr = np.empty((n, 1), dtype=np.float64)
    cdef cnp.ndarray[double, ndim=2] lhs_a_arr = np.empty((n, n), dtype=np.float64)
    cdef cnp.ndarray[double, ndim=2] lhs_d_arr = np.empty((n, n), dtype=np.float64)
    cdef cnp.ndarray[double, ndim=2] lhs_t_arr = np.empty((n, n), dtype=np.float64)
    cdef cnp.ndarray[double, ndim=2] lhs_l_arr = np.empty((n, n), dtype=np.float64)
    cdef cnp.ndarray[double, ndim=2] u_arr = np.empty((n, 1), dtype=np.float64)
    cdef cnp.ndarray[double, ndim=2] v_arr = np.empty((n, 1), dtype=np.float64)
    cdef const double[:, ::1] A00 = A00_arr
    cdef const double[:, ::1] D33 = D33_arr
    cdef const double[:, ::1] Aeq = Aeq_arr
    cdef const double[:, ::1] Top = Top_arr
    cdef const double[:, ::1] Linv_rhs = Linv_rhs_arr
    cdef const double[:, ::1] Linv_X = Linv_X_arr
    cdef const double[::1] inv = inv_arr
    cdef const double[:, ::1] rhs_p = rhs_p_arr
    cdef const double[:, ::1] rhs_d = rhs_d_arr
    cdef const double[:, ::1] rhs_t = rhs_t_arr
    cdef const double[:, ::1] Aeq_T = Aeq_T_arr
    cdef double[:, ::1] scaled = scaled_arr
    cdef double[:, ::1] tmp_mat = tmp_mat_arr
    cdef double[:, ::1] tmp_rhs = tmp_rhs_arr
    cdef double[:, ::1] lhs_a = lhs_a_arr
    cdef double[:, ::1] lhs_d = lhs_d_arr
    cdef double[:, ::1] lhs_t = lhs_t_arr
    cdef double[:, ::1] lhs_l = lhs_l_arr
    cdef double[:, ::1] u = u_arr
    cdef double[:, ::1] v = v_arr
    cdef int i, j
    with nogil:
        for i in range(n):
            for j in range(n):
                scaled[i, j] = Linv_X[i, j] * inv[j]
        cy_dgemm(scaled, rhs_d, tmp_rhs)
        for i in range(n):
            tmp_rhs[i, 0] = Linv_rhs[i, 0] - tmp_rhs[i, 0]
        cy_dgemm(Aeq, tmp_rhs, u)
        cy_dgemm(Top, tmp_rhs, v)
        for i in range(n):
            u[i, 0] = rhs_p[i, 0] - u[i, 0]
            v[i, 0] = rhs_t[i, 0] - v[i, 0]
        cy_dgemm(Aeq, scaled, tmp_mat)
        cy_dgemm(tmp_mat, Aeq_T, lhs_a)
        for i in range(n):
            for j in range(n):
                lhs_a[i, j] += A00[i, j]
        cy_dgemm(Top, Linv_X, lhs_d)
        for i in range(n):
            for j in range(n):
                lhs_d[i, j] += D33[i, j]
        cy_dgemm(Top, scaled, tmp_mat)
        cy_dgemm(tmp_mat, Aeq_T, lhs_t)
        cy_dgemm(Aeq, Linv_X, lhs_l)
    return lhs_a_arr, lhs_d_arr, lhs_t_arr, lhs_l_arr, u_arr, v_arr


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
cdef void einsum_general(
        const double[:, ::1] XAX,
        const double[:, ::1] block_A,
        const double[:, ::1] XAX1,
        const double[:, ::1] x_core,
        double[:, ::1] out,
        double[:, ::1] intermediate_mat1,
        double[:, ::1] intermediate_mat1_2,
        double[:, ::1] intermediate_mat2,
        double[:, ::1] intermediate_mat2_2,
        int r_in,
        int n_in,
        int R_in,
        int l_out,
        int m_out,
        int L_out,
        double alpha,
        double beta
) noexcept nogil:
    cdef:
        int S = int(block_A.shape[0] / n_in)
        int s = int(block_A.shape[1] / m_out)

    cy_dgemm(x_core, XAX1, intermediate_mat1)
    _transpose_reshape_step2(intermediate_mat1, intermediate_mat1_2, r_in, L_out, n_in, S)
    cy_dgemm(intermediate_mat1_2, block_A, intermediate_mat2)
    _transpose_reshape_step3(intermediate_mat2, intermediate_mat2_2, r_in, L_out, s, m_out)
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


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.inline
cdef void pack_result_general(double[:, :] result, double[:] flat_result, int L, int m, int l) noexcept nogil:
    cdef int li, mi, Li
    cdef int idx

    for li in range(l):
        for mi in range(m):
            for Li in range(L):
                idx = (li * m + mi) * L + Li
                flat_result[idx] = result[Li * m + mi, li]


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
    cdef void _fill_block(self, int n, int p, double[:, ::1] out) noexcept nogil:
        cdef const double[:, :, ::1] XAX_by_r = self.XAX_by_r
        cdef const double[:, :, ::1] A0_by_n = self.A0_by_n
        cdef const double[:, :, ::1] A1_by_p = self.A1_by_p
        cdef const double[:, :, ::1] XAX2_by_R_T = self.XAX2_by_R_T
        cdef double[:, ::1] coeff = self.coeff
        cdef double[:, ::1] tmp = self.tmp
        cdef double[:, ::1] small = self.small
        cdef int li, Li, ri, Ri
        cy_dgemm(A0_by_n[n, :, :], A1_by_p[p, :, :], coeff)
        for ri in range(self.rdim):
            cy_dgemm(XAX_by_r[ri, :, :], coeff, tmp)
            for Ri in range(self.Rdim):
                cy_dgemm(tmp, XAX2_by_R_T[Ri, :, :], small)
                for li in range(self.ldim):
                    for Li in range(self.Ldim):
                        out[li * self.Ldim + Li, ri * self.Rdim + Ri] = small[li, Li]

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.nonecheck(False)
    cpdef cnp.ndarray[double, ndim=2] block(self, int n, int p):
        cdef cnp.ndarray[double, ndim=2] out_arr = np.empty(
            (self.ldim * self.Ldim, self.rdim * self.Rdim), dtype=np.float64
        )
        cdef double[:, ::1] out = out_arr
        with nogil:
            self._fill_block(n, p, out)
        return out_arr

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.nonecheck(False)
    cpdef object best_block_eig(self, cnp.ndarray[double, ndim=1] previous, double eps, int max_dense=128):
        cdef int m = self.rdim * self.Rdim
        cdef int work_size = 3 * m - 1
        cdef cnp.ndarray[double, ndim=1] previous_arr
        cdef cnp.ndarray[double, ndim=2] block_arr
        cdef cnp.ndarray[double, ndim=1] eigvals_arr
        cdef cnp.ndarray[double, ndim=1] work_arr
        cdef cnp.ndarray[double, ndim=1] best_vec_arr
        cdef cnp.ndarray[double, ndim=1] solution_arr
        cdef const double[::1] previous_v
        cdef double[:, ::1] block
        cdef double[::1] eigvals, work, best_vec, solution
        cdef double val = 0.0
        cdef double best_val = 0.0
        cdef double warm_norm = 0.0
        cdef double best_warm_norm = -1.0
        cdef double entry
        cdef bint has_best = False
        cdef bint failed = False
        cdef int info = 0
        cdef int n, p, ri, Ri, idx, best_n = 0, best_p = 0
        if self.ldim != self.rdim or self.Ldim != self.Rdim or m > max_dense:
            return None
        if work_size < 1:
            work_size = 1
        previous_arr = np.ascontiguousarray(previous, dtype=np.float64)
        if previous_arr.shape[0] != self.in_size:
            return None
        block_arr = np.empty((m, m), dtype=np.float64)
        eigvals_arr = np.empty(m, dtype=np.float64)
        work_arr = np.empty(work_size, dtype=np.float64)
        best_vec_arr = np.empty(m, dtype=np.float64)
        solution_arr = np.zeros(self.in_size, dtype=np.float64)
        previous_v = previous_arr
        block = block_arr
        eigvals = eigvals_arr
        work = work_arr
        best_vec = best_vec_arr
        solution = solution_arr
        with nogil:
            for n in range(self.n0):
                for p in range(self.n1):
                    self._fill_block(n, p, block)
                    info = _sym_smallest_eigh(block, eigvals, work, &val)
                    if info != 0:
                        failed = True
                        break
                    warm_norm = 0.0
                    for ri in range(self.rdim):
                        for Ri in range(self.Rdim):
                            entry = previous_v[((ri * self.n0 + n) * self.n1 + p) * self.Rdim + Ri]
                            warm_norm += entry * entry
                    warm_norm = sqrt(warm_norm)
                    if _prefer_block_c(val, warm_norm, best_val, best_warm_norm, has_best, eps):
                        has_best = True
                        best_val = val
                        best_warm_norm = warm_norm
                        best_n = n
                        best_p = p
                        for idx in range(m):
                            best_vec[idx] = block[0, idx]
                if failed:
                    break
            if not failed:
                for ri in range(self.rdim):
                    for Ri in range(self.Rdim):
                        idx = ri * self.Rdim + Ri
                        solution[((ri * self.n0 + best_n) * self.n1 + best_p) * self.Rdim + Ri] = best_vec[idx]
        if failed:
            return None
        return best_val, solution_arr

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
    cdef void _fill_block(self, int n, double[:, ::1] out) noexcept nogil:
        cdef const double[:, :, ::1] XAX_by_r = self.XAX_by_r
        cdef const double[:, :, ::1] A0_by_n = self.A0_by_n
        cdef const double[:, :, ::1] XAX1_by_R_T = self.XAX1_by_R_T
        cdef double[:, ::1] tmp = self.tmp
        cdef double[:, ::1] small = self.small
        cdef int li, Li, ri, Ri
        for ri in range(self.rdim):
            cy_dgemm(XAX_by_r[ri, :, :], A0_by_n[n, :, :], tmp)
            for Ri in range(self.Rdim):
                cy_dgemm(tmp, XAX1_by_R_T[Ri, :, :], small)
                for li in range(self.ldim):
                    for Li in range(self.Ldim):
                        out[li * self.Ldim + Li, ri * self.Rdim + Ri] = small[li, Li]

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.nonecheck(False)
    cpdef cnp.ndarray[double, ndim=2] block(self, int n):
        cdef cnp.ndarray[double, ndim=2] out_arr = np.empty(
            (self.ldim * self.Ldim, self.rdim * self.Rdim), dtype=np.float64
        )
        cdef double[:, ::1] out = out_arr
        with nogil:
            self._fill_block(n, out)
        return out_arr

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.nonecheck(False)
    cpdef object best_block_eig(self, cnp.ndarray[double, ndim=1] previous, double eps, int max_dense=128):
        cdef int m = self.rdim * self.Rdim
        cdef int work_size = 3 * m - 1
        cdef cnp.ndarray[double, ndim=1] previous_arr
        cdef cnp.ndarray[double, ndim=2] block_arr
        cdef cnp.ndarray[double, ndim=1] eigvals_arr
        cdef cnp.ndarray[double, ndim=1] work_arr
        cdef cnp.ndarray[double, ndim=1] best_vec_arr
        cdef cnp.ndarray[double, ndim=1] solution_arr
        cdef const double[::1] previous_v
        cdef double[:, ::1] block
        cdef double[::1] eigvals, work, best_vec, solution
        cdef double val = 0.0
        cdef double best_val = 0.0
        cdef double warm_norm = 0.0
        cdef double best_warm_norm = -1.0
        cdef double entry
        cdef bint has_best = False
        cdef bint failed = False
        cdef int info = 0
        cdef int n, ri, Ri, idx, best_n = 0
        if self.ldim != self.rdim or self.Ldim != self.Rdim or m > max_dense:
            return None
        if work_size < 1:
            work_size = 1
        previous_arr = np.ascontiguousarray(previous, dtype=np.float64)
        if previous_arr.shape[0] != self.in_size:
            return None
        block_arr = np.empty((m, m), dtype=np.float64)
        eigvals_arr = np.empty(m, dtype=np.float64)
        work_arr = np.empty(work_size, dtype=np.float64)
        best_vec_arr = np.empty(m, dtype=np.float64)
        solution_arr = np.zeros(self.in_size, dtype=np.float64)
        previous_v = previous_arr
        block = block_arr
        eigvals = eigvals_arr
        work = work_arr
        best_vec = best_vec_arr
        solution = solution_arr
        with nogil:
            for n in range(self.n0):
                self._fill_block(n, block)
                info = _sym_smallest_eigh(block, eigvals, work, &val)
                if info != 0:
                    failed = True
                    break
                warm_norm = 0.0
                for ri in range(self.rdim):
                    for Ri in range(self.Rdim):
                        entry = previous_v[(ri * self.n0 + n) * self.Rdim + Ri]
                        warm_norm += entry * entry
                warm_norm = sqrt(warm_norm)
                if _prefer_block_c(val, warm_norm, best_val, best_warm_norm, has_best, eps):
                    has_best = True
                    best_val = val
                    best_warm_norm = warm_norm
                    best_n = n
                    for idx in range(m):
                        best_vec[idx] = block[0, idx]
            if not failed:
                for ri in range(self.rdim):
                    for Ri in range(self.Rdim):
                        idx = ri * self.Rdim + Ri
                        solution[(ri * self.n0 + best_n) * self.Rdim + Ri] = best_vec[idx]
        if failed:
            return None
        return best_val, solution_arr

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


cdef class CoreMatVecWrapper:
    cdef double[:, ::1] result, workspace1, workspace1_2, workspace2, workspace2_2
    cdef object flat_result_arr
    cdef double[:] flat_result
    cdef const double[:, ::1] XAX, block_A, XAX1
    cdef int r, n, R, l, m, L, total_size

    def __init__(
            self,
            cnp.ndarray[double, ndim=3] Phi_l,
            cnp.ndarray[double, ndim=4] A_k,
            cnp.ndarray[double, ndim=3] Phi_r
    ):
        self.r = Phi_l.shape[2]
        self.n = A_k.shape[2]
        self.R = Phi_r.shape[2]
        self.l = Phi_l.shape[0]
        self.m = A_k.shape[1]
        self.L = Phi_r.shape[0]
        self.total_size = self.l * self.m * self.L

        self.XAX = np.ascontiguousarray(Phi_l.transpose(0, 2, 1).reshape(self.l, -1).T)
        self.block_A = np.ascontiguousarray(A_k.reshape(A_k.shape[0] * self.m, self.n * A_k.shape[3]).T)
        self.XAX1 = np.ascontiguousarray(Phi_r.reshape(-1, self.R).T)

        self.result = np.empty((self.L * self.m, self.l), dtype=np.float64)
        self.workspace1 = np.empty((self.r * self.n, self.L * A_k.shape[3]), dtype=np.float64)
        self.workspace1_2 = np.empty((self.r * self.L, self.n * A_k.shape[3]), dtype=np.float64)
        self.workspace2 = np.empty((self.r * self.L, A_k.shape[0] * self.m), dtype=np.float64)
        self.workspace2_2 = np.empty((self.L * self.m, self.r * A_k.shape[0]), dtype=np.float64)
        self.flat_result_arr = np.empty(self.total_size, dtype=np.float64)
        self.flat_result = self.flat_result_arr

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.nonecheck(False)
    cpdef cnp.ndarray[double, ndim=1] matvec(self, cnp.ndarray[double, ndim=1] x_core):
        cdef const double[:, ::1] x_view = x_core.reshape(self.r * self.n, self.R)
        with nogil:
            einsum_general(
                self.XAX, self.block_A, self.XAX1, x_view, self.result,
                self.workspace1, self.workspace1_2, self.workspace2, self.workspace2_2,
                self.r, self.n, self.R, self.l, self.m, self.L, 1.0, 0.0
            )
            pack_result_general(self.result, self.flat_result, self.L, self.m, self.l)
        return self.flat_result_arr


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
cpdef cnp.ndarray[double, ndim=3] core_matvec(
        cnp.ndarray[double, ndim=3] Phi_l,
        cnp.ndarray[double, ndim=4] A_k,
        cnp.ndarray[double, ndim=3] Phi_r,
        cnp.ndarray[double, ndim=3] x_core
):
    cdef CoreMatVecWrapper helper = CoreMatVecWrapper(Phi_l, A_k, Phi_r)
    cdef cnp.ndarray[double, ndim=1] x_flat = np.ascontiguousarray(x_core.reshape(-1), dtype=np.float64)
    return helper.matvec(x_flat).reshape(Phi_l.shape[0], A_k.shape[1], Phi_r.shape[0])


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
cpdef cnp.ndarray[double, ndim=3] core_rmatvec(
        cnp.ndarray[double, ndim=3] Phi_l,
        cnp.ndarray[double, ndim=4] A_k,
        cnp.ndarray[double, ndim=3] Phi_r,
        cnp.ndarray[double, ndim=3] x_core
):
    cdef cnp.ndarray[double, ndim=3] Phi_l_T = np.ascontiguousarray(np.transpose(Phi_l, axes=(2, 1, 0)))
    cdef cnp.ndarray[double, ndim=4] A_k_T = np.ascontiguousarray(np.transpose(A_k, axes=(0, 2, 1, 3)))
    cdef cnp.ndarray[double, ndim=3] Phi_r_T = np.ascontiguousarray(np.transpose(Phi_r, axes=(2, 1, 0)))
    return core_matvec(Phi_l_T, A_k_T, Phi_r_T, x_core)


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
cpdef cnp.ndarray[double, ndim=3] rhs_contract(
        cnp.ndarray[double, ndim=2] Phi_l,
        cnp.ndarray[double, ndim=3] b_core,
        cnp.ndarray[double, ndim=2] Phi_r,
        double scale=1.0
):
    cdef int bdim = Phi_l.shape[0]
    cdef int rdim = Phi_l.shape[1]
    cdef int mdim = b_core.shape[1]
    cdef int Bdim = Phi_r.shape[0]
    cdef int Rdim = Phi_r.shape[1]
    cdef cnp.ndarray[double, ndim=2] Phi_l_T_arr = np.ascontiguousarray(Phi_l.T)
    cdef cnp.ndarray[double, ndim=3] b_arr = np.ascontiguousarray(np.transpose(b_core, (1, 0, 2)))
    cdef cnp.ndarray[double, ndim=2] Phi_r_arr = np.ascontiguousarray(Phi_r)
    cdef cnp.ndarray[double, ndim=2] tmp_arr = np.empty((rdim, Bdim), dtype=np.float64)
    cdef cnp.ndarray[double, ndim=2] small_arr = np.empty((rdim, Rdim), dtype=np.float64)
    cdef cnp.ndarray[double, ndim=3] out_arr = np.empty((rdim, mdim, Rdim), dtype=np.float64)
    cdef const double[:, ::1] Phi_l_T = Phi_l_T_arr
    cdef const double[:, :, ::1] b_by_m = b_arr
    cdef const double[:, ::1] Phi_r_v = Phi_r_arr
    cdef double[:, ::1] tmp = tmp_arr
    cdef double[:, ::1] small = small_arr
    cdef double[:, :, ::1] out = out_arr
    cdef int mi, ri, Ri
    with nogil:
        for mi in range(mdim):
            cy_dgemm(Phi_l_T, b_by_m[mi, :, :], tmp)
            cy_dgemm(tmp, Phi_r_v, small, scale, 0.0)
            for ri in range(rdim):
                for Ri in range(Rdim):
                    out[ri, mi, Ri] = small[ri, Ri]
    return out_arr


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
cpdef cnp.ndarray[double, ndim=2] dense_core_matrix(
        cnp.ndarray[double, ndim=3] Phi_l,
        cnp.ndarray[double, ndim=4] A_k,
        cnp.ndarray[double, ndim=3] Phi_r
):
    cdef int ldim = Phi_l.shape[0]
    cdef int sdim = Phi_l.shape[1]
    cdef int rdim = Phi_l.shape[2]
    cdef int mdim = A_k.shape[1]
    cdef int ndim = A_k.shape[2]
    cdef int Sdim = A_k.shape[3]
    cdef int Ldim = Phi_r.shape[0]
    cdef int Rdim = Phi_r.shape[2]
    cdef cnp.ndarray[double, ndim=3] left_by_r_arr = np.ascontiguousarray(np.transpose(Phi_l, (2, 0, 1)))
    cdef cnp.ndarray[double, ndim=4] A_by_mn_arr = np.ascontiguousarray(np.transpose(A_k, (1, 2, 0, 3)))
    cdef cnp.ndarray[double, ndim=3] right_by_R_T_arr = np.ascontiguousarray(np.transpose(Phi_r, (2, 1, 0)))
    cdef cnp.ndarray[double, ndim=2] tmp_arr = np.empty((ldim, Sdim), dtype=np.float64)
    cdef cnp.ndarray[double, ndim=2] small_arr = np.empty((ldim, Ldim), dtype=np.float64)
    cdef cnp.ndarray[double, ndim=2] out_arr = np.empty((ldim * mdim * Ldim, rdim * ndim * Rdim), dtype=np.float64)
    cdef const double[:, :, ::1] left_by_r = left_by_r_arr
    cdef const double[:, :, :, ::1] A_by_mn = A_by_mn_arr
    cdef const double[:, :, ::1] right_by_R_T = right_by_R_T_arr
    cdef double[:, ::1] tmp = tmp_arr
    cdef double[:, ::1] small = small_arr
    cdef double[:, ::1] out = out_arr
    cdef int mi, ni, ri, Ri, li, Li, row_base, col_idx
    with nogil:
        for mi in range(mdim):
            for ni in range(ndim):
                for ri in range(rdim):
                    cy_dgemm(left_by_r[ri, :, :], A_by_mn[mi, ni, :, :], tmp)
                    for Ri in range(Rdim):
                        cy_dgemm(tmp, right_by_R_T[Ri, :, :], small)
                        col_idx = (ri * ndim + ni) * Rdim + Ri
                        for li in range(ldim):
                            row_base = (li * mdim + mi) * Ldim
                            for Li in range(Ldim):
                                out[row_base + Li, col_idx] = small[li, Li]
    return out_arr


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
cpdef cnp.ndarray[double, ndim=2] dense_two_core_matrix(
        cnp.ndarray[double, ndim=3] Phi_l,
        cnp.ndarray[double, ndim=4] A_l,
        cnp.ndarray[double, ndim=4] A_r,
        cnp.ndarray[double, ndim=3] Phi_r
):
    cdef int ldim = Phi_l.shape[0]
    cdef int sdim = Phi_l.shape[1]
    cdef int rdim = Phi_l.shape[2]
    cdef int mdim = A_l.shape[1]
    cdef int ndim = A_l.shape[2]
    cdef int kdim = A_l.shape[3]
    cdef int pdim = A_r.shape[1]
    cdef int tdim = A_r.shape[2]
    cdef int Sdim = A_r.shape[3]
    cdef int Ldim = Phi_r.shape[0]
    cdef int Rdim = Phi_r.shape[2]
    cdef cnp.ndarray[double, ndim=3] left_by_r_arr = np.ascontiguousarray(np.transpose(Phi_l, (2, 0, 1)))
    cdef cnp.ndarray[double, ndim=4] A_l_by_mn_arr = np.ascontiguousarray(np.transpose(A_l, (1, 2, 0, 3)))
    cdef cnp.ndarray[double, ndim=4] A_r_by_pt_arr = np.ascontiguousarray(np.transpose(A_r, (1, 2, 0, 3)))
    cdef cnp.ndarray[double, ndim=3] right_by_R_T_arr = np.ascontiguousarray(np.transpose(Phi_r, (2, 1, 0)))
    cdef cnp.ndarray[double, ndim=2] coeff_arr = np.empty((sdim, Sdim), dtype=np.float64)
    cdef cnp.ndarray[double, ndim=2] tmp_arr = np.empty((ldim, Sdim), dtype=np.float64)
    cdef cnp.ndarray[double, ndim=2] small_arr = np.empty((ldim, Ldim), dtype=np.float64)
    cdef cnp.ndarray[double, ndim=2] out_arr = np.empty(
        (ldim * mdim * pdim * Ldim, rdim * ndim * tdim * Rdim),
        dtype=np.float64
    )
    cdef const double[:, :, ::1] left_by_r = left_by_r_arr
    cdef const double[:, :, :, ::1] A_l_by_mn = A_l_by_mn_arr
    cdef const double[:, :, :, ::1] A_r_by_pt = A_r_by_pt_arr
    cdef const double[:, :, ::1] right_by_R_T = right_by_R_T_arr
    cdef double[:, ::1] coeff = coeff_arr
    cdef double[:, ::1] tmp = tmp_arr
    cdef double[:, ::1] small = small_arr
    cdef double[:, ::1] out = out_arr
    cdef int mi, pi, ni, ti, ri, Ri, li, Li, row_base, col_idx
    with nogil:
        for mi in range(mdim):
            for pi in range(pdim):
                for ni in range(ndim):
                    for ti in range(tdim):
                        cy_dgemm(A_l_by_mn[mi, ni, :, :], A_r_by_pt[pi, ti, :, :], coeff)
                        for ri in range(rdim):
                            cy_dgemm(left_by_r[ri, :, :], coeff, tmp)
                            for Ri in range(Rdim):
                                cy_dgemm(tmp, right_by_R_T[Ri, :, :], small)
                                col_idx = ((ri * ndim + ni) * tdim + ti) * Rdim + Ri
                                for li in range(ldim):
                                    row_base = ((li * mdim + mi) * pdim + pi) * Ldim
                                    for Li in range(Ldim):
                                        out[row_base + Li, col_idx] = small[li, Li]
    return out_arr


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
cpdef cnp.ndarray[double, ndim=3] core_sum_contract(
        cnp.ndarray[double, ndim=3] Phi_l,
        cnp.ndarray[double, ndim=4] A_k,
        cnp.ndarray[double, ndim=3] Phi_r
):
    cdef int ldim = Phi_l.shape[0]
    cdef int mdim = A_k.shape[1]
    cdef int Sdim = A_k.shape[3]
    cdef int Ldim = Phi_r.shape[0]
    cdef cnp.ndarray[double, ndim=2] left_arr = np.ascontiguousarray(np.sum(Phi_l, axis=2))
    cdef cnp.ndarray[double, ndim=3] A_by_m_arr = np.ascontiguousarray(np.transpose(np.sum(A_k, axis=2), (1, 0, 2)))
    cdef cnp.ndarray[double, ndim=2] right_T_arr = np.ascontiguousarray(np.sum(Phi_r, axis=2).T)
    cdef cnp.ndarray[double, ndim=2] tmp_arr = np.empty((ldim, Sdim), dtype=np.float64)
    cdef cnp.ndarray[double, ndim=2] small_arr = np.empty((ldim, Ldim), dtype=np.float64)
    cdef cnp.ndarray[double, ndim=3] out_arr = np.empty((ldim, mdim, Ldim), dtype=np.float64)
    cdef const double[:, ::1] left = left_arr
    cdef const double[:, :, ::1] A_by_m = A_by_m_arr
    cdef const double[:, ::1] right_T = right_T_arr
    cdef double[:, ::1] tmp = tmp_arr
    cdef double[:, ::1] small = small_arr
    cdef double[:, :, ::1] out = out_arr
    cdef int mi, li, Li
    with nogil:
        for mi in range(mdim):
            cy_dgemm(left, A_by_m[mi, :, :], tmp)
            cy_dgemm(tmp, right_T, small)
            for li in range(ldim):
                for Li in range(Ldim):
                    out[li, mi, Li] = small[li, Li]
    return out_arr


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
cpdef cnp.ndarray[double, ndim=4] matmat_product_core(
        cnp.ndarray[double, ndim=3] Phi_l,
        cnp.ndarray[double, ndim=4] A_k,
        cnp.ndarray[double, ndim=4] D_k,
        cnp.ndarray[double, ndim=3] Phi_r
):
    cdef int rdim = Phi_l.shape[0]
    cdef int adim = Phi_l.shape[1]
    cdef int bdim = Phi_l.shape[2]
    cdef int mdim = A_k.shape[1]
    cdef int kdim = A_k.shape[2]
    cdef int Adim = A_k.shape[3]
    cdef int ndim = D_k.shape[2]
    cdef int Bdim = D_k.shape[3]
    cdef int Rdim = Phi_r.shape[0]
    cdef cnp.ndarray[double, ndim=2] left_flat_arr = np.ascontiguousarray(Phi_l.reshape(rdim, adim * bdim))
    cdef cnp.ndarray[double, ndim=4] A_by_mk_T_arr = np.ascontiguousarray(np.transpose(A_k, (1, 2, 3, 0)))
    cdef cnp.ndarray[double, ndim=4] D_by_kn_arr = np.ascontiguousarray(np.transpose(D_k, (1, 2, 0, 3)))
    cdef cnp.ndarray[double, ndim=3] right_by_R_T_arr = np.ascontiguousarray(np.transpose(Phi_r, (0, 2, 1)))
    cdef cnp.ndarray[double, ndim=2] d_right_arr = np.empty((bdim, Adim), dtype=np.float64)
    cdef cnp.ndarray[double, ndim=2] q_ba_arr = np.empty((bdim, adim), dtype=np.float64)
    cdef cnp.ndarray[double, ndim=2] q_flat_arr = np.empty((adim * bdim, Rdim), dtype=np.float64)
    cdef cnp.ndarray[double, ndim=2] small_arr = np.empty((rdim, Rdim), dtype=np.float64)
    cdef cnp.ndarray[double, ndim=4] out_arr = np.empty((rdim, mdim, ndim, Rdim), dtype=np.float64)
    cdef const double[:, ::1] left_flat = left_flat_arr
    cdef const double[:, :, :, ::1] A_by_mk_T = A_by_mk_T_arr
    cdef const double[:, :, :, ::1] D_by_kn = D_by_kn_arr
    cdef const double[:, :, ::1] right_by_R_T = right_by_R_T_arr
    cdef double[:, ::1] d_right = d_right_arr
    cdef double[:, ::1] q_ba = q_ba_arr
    cdef double[:, ::1] q_flat = q_flat_arr
    cdef double[:, ::1] small = small_arr
    cdef double[:, :, :, ::1] out = out_arr
    cdef int mi, ni, ki, Ri, ai, bi, ri
    cdef double beta
    with nogil:
        for mi in range(mdim):
            for ni in range(ndim):
                beta = 0.0
                for ki in range(kdim):
                    for Ri in range(Rdim):
                        cy_dgemm(D_by_kn[ki, ni, :, :], right_by_R_T[Ri, :, :], d_right)
                        cy_dgemm(d_right, A_by_mk_T[mi, ki, :, :], q_ba)
                        for ai in range(adim):
                            for bi in range(bdim):
                                q_flat[ai * bdim + bi, Ri] = q_ba[bi, ai]
                    cy_dgemm(left_flat, q_flat, small, 1.0, beta)
                    beta = 1.0
                for ri in range(rdim):
                    for Ri in range(Rdim):
                        out[ri, mi, ni, Ri] = small[ri, Ri]
    return out_arr


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
cpdef cnp.ndarray[double, ndim=3] matmat_phi_bck(
        cnp.ndarray[double, ndim=3] Phi_r,
        cnp.ndarray[double, ndim=4] A_k,
        cnp.ndarray[double, ndim=4] D_k,
        cnp.ndarray[double, ndim=4] x_core
):
    cdef int Rdim = Phi_r.shape[0]
    cdef int Adim = Phi_r.shape[1]
    cdef int Bdim = Phi_r.shape[2]
    cdef int adim = A_k.shape[0]
    cdef int mdim = A_k.shape[1]
    cdef int kdim = A_k.shape[2]
    cdef int bdim = D_k.shape[0]
    cdef int ndim = D_k.shape[2]
    cdef int rdim = x_core.shape[0]
    cdef cnp.ndarray[double, ndim=2] Phi_flat_arr = np.ascontiguousarray(Phi_r.reshape(Rdim, Adim * Bdim))
    cdef cnp.ndarray[double, ndim=4] A_by_mk_T_arr = np.ascontiguousarray(np.transpose(A_k, (1, 2, 3, 0)))
    cdef cnp.ndarray[double, ndim=4] D_by_kn_arr = np.ascontiguousarray(np.transpose(D_k, (1, 2, 0, 3)))
    cdef cnp.ndarray[double, ndim=4] x_by_mn_arr = np.ascontiguousarray(np.transpose(x_core, (1, 2, 0, 3)))
    cdef cnp.ndarray[double, ndim=2] T_arr = np.empty((rdim, Adim * Bdim), dtype=np.float64)
    cdef cnp.ndarray[double, ndim=2] T_by_rA_arr = T_arr.reshape((rdim * Adim, Bdim))
    cdef cnp.ndarray[double, ndim=2] D_T_arr = np.empty((Bdim, bdim), dtype=np.float64)
    cdef cnp.ndarray[double, ndim=2] tmp_all_arr = np.empty((rdim * Adim, bdim), dtype=np.float64)
    cdef cnp.ndarray[double, ndim=2] tmp_mat_arr = np.empty((rdim, Adim), dtype=np.float64)
    cdef cnp.ndarray[double, ndim=2] small_arr = np.empty((rdim, adim), dtype=np.float64)
    cdef cnp.ndarray[double, ndim=3] out_arr = np.zeros((rdim, adim, bdim), dtype=np.float64)
    cdef const double[:, ::1] Phi_flat = Phi_flat_arr
    cdef const double[:, :, :, ::1] A_by_mk_T = A_by_mk_T_arr
    cdef const double[:, :, :, ::1] D_by_kn = D_by_kn_arr
    cdef const double[:, :, :, ::1] x_by_mn = x_by_mn_arr
    cdef double[:, ::1] T = T_arr
    cdef double[:, ::1] T_by_rA = T_by_rA_arr
    cdef double[:, ::1] D_T = D_T_arr
    cdef double[:, ::1] tmp_all = tmp_all_arr
    cdef double[:, ::1] tmp_mat = tmp_mat_arr
    cdef double[:, ::1] small = small_arr
    cdef double[:, :, ::1] out = out_arr
    cdef int mi, ni, ki, bi, ri, ai, Ai, Bi
    with nogil:
        for mi in range(mdim):
            for ni in range(ndim):
                cy_dgemm(x_by_mn[mi, ni, :, :], Phi_flat, T)
                for ki in range(kdim):
                    for Bi in range(Bdim):
                        for bi in range(bdim):
                            D_T[Bi, bi] = D_by_kn[ki, ni, bi, Bi]
                    cy_dgemm(T_by_rA, D_T, tmp_all)
                    for bi in range(bdim):
                        for ri in range(rdim):
                            for Ai in range(Adim):
                                tmp_mat[ri, Ai] = tmp_all[ri * Adim + Ai, bi]
                        cy_dgemm(tmp_mat, A_by_mk_T[mi, ki, :, :], small)
                        for ri in range(rdim):
                            for ai in range(adim):
                                out[ri, ai, bi] += small[ri, ai]
    return out_arr


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
cpdef cnp.ndarray[double, ndim=3] matmat_phi_fwd(
        cnp.ndarray[double, ndim=3] Phi_l,
        cnp.ndarray[double, ndim=4] A_k,
        cnp.ndarray[double, ndim=4] D_k,
        cnp.ndarray[double, ndim=4] x_core
):
    cdef int rdim = Phi_l.shape[0]
    cdef int adim = Phi_l.shape[1]
    cdef int bdim = Phi_l.shape[2]
    cdef int mdim = A_k.shape[1]
    cdef int kdim = A_k.shape[2]
    cdef int Adim = A_k.shape[3]
    cdef int ndim = D_k.shape[2]
    cdef int Bdim = D_k.shape[3]
    cdef int Rdim = x_core.shape[3]
    cdef cnp.ndarray[double, ndim=2] left_flat_arr = np.ascontiguousarray(Phi_l.reshape(rdim, adim * bdim))
    cdef cnp.ndarray[double, ndim=4] A_by_mk_arr = np.ascontiguousarray(np.transpose(A_k, (1, 2, 0, 3)))
    cdef cnp.ndarray[double, ndim=4] D_by_kn_arr = np.ascontiguousarray(np.transpose(D_k, (1, 2, 0, 3)))
    cdef cnp.ndarray[double, ndim=4] x_by_mn_T_arr = np.ascontiguousarray(np.transpose(x_core, (1, 2, 3, 0)))
    cdef cnp.ndarray[double, ndim=2] T_arr = np.empty((Rdim, adim * bdim), dtype=np.float64)
    cdef cnp.ndarray[double, ndim=2] T_ba_arr = np.empty((Rdim * bdim, adim), dtype=np.float64)
    cdef cnp.ndarray[double, ndim=2] U_all_arr = np.empty((Rdim * bdim, Adim), dtype=np.float64)
    cdef cnp.ndarray[double, ndim=2] U_arr = np.empty((Rdim, bdim), dtype=np.float64)
    cdef cnp.ndarray[double, ndim=2] small_arr = np.empty((Rdim, Bdim), dtype=np.float64)
    cdef cnp.ndarray[double, ndim=3] out_arr = np.zeros((Rdim, Adim, Bdim), dtype=np.float64)
    cdef const double[:, ::1] left_flat = left_flat_arr
    cdef const double[:, :, :, ::1] A_by_mk = A_by_mk_arr
    cdef const double[:, :, :, ::1] D_by_kn = D_by_kn_arr
    cdef const double[:, :, :, ::1] x_by_mn_T = x_by_mn_T_arr
    cdef double[:, ::1] T = T_arr
    cdef double[:, ::1] T_ba = T_ba_arr
    cdef double[:, ::1] U_all = U_all_arr
    cdef double[:, ::1] U = U_arr
    cdef double[:, ::1] small = small_arr
    cdef double[:, :, ::1] out = out_arr
    cdef int mi, ni, ki, Ai, ai, bi, Ri, Bi
    with nogil:
        for mi in range(mdim):
            for ni in range(ndim):
                cy_dgemm(x_by_mn_T[mi, ni, :, :], left_flat, T)
                for Ri in range(Rdim):
                    for bi in range(bdim):
                        for ai in range(adim):
                            T_ba[Ri * bdim + bi, ai] = T[Ri, ai * bdim + bi]
                for ki in range(kdim):
                    cy_dgemm(T_ba, A_by_mk[mi, ki, :, :], U_all)
                    for Ai in range(Adim):
                        for Ri in range(Rdim):
                            for bi in range(bdim):
                                U[Ri, bi] = U_all[Ri * bdim + bi, Ai]
                        cy_dgemm(U, D_by_kn[ki, ni, :, :], small)
                        for Ri in range(Rdim):
                            for Bi in range(Bdim):
                                out[Ri, Ai, Bi] += small[Ri, Bi]
    return out_arr


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
cpdef cnp.ndarray[double, ndim=2] phi_bck_rhs(
        cnp.ndarray[double, ndim=2] Phi_now,
        cnp.ndarray[double, ndim=3] b_core,
        cnp.ndarray[double, ndim=3] core
):
    cdef int bdim = b_core.shape[0]
    cdef int ndim = b_core.shape[1]
    cdef int Bdim = b_core.shape[2]
    cdef int rdim = core.shape[0]
    cdef int Rdim = core.shape[2]
    cdef cnp.ndarray[double, ndim=3] b_by_n_arr = np.ascontiguousarray(np.transpose(b_core, (1, 0, 2)))
    cdef cnp.ndarray[double, ndim=3] core_by_n_T_arr = np.ascontiguousarray(np.transpose(core, (1, 2, 0)))
    cdef cnp.ndarray[double, ndim=2] Phi_arr = np.ascontiguousarray(Phi_now)
    cdef cnp.ndarray[double, ndim=2] tmp_arr = np.empty((bdim, Rdim), dtype=np.float64)
    cdef cnp.ndarray[double, ndim=2] out_arr = np.zeros((bdim, rdim), dtype=np.float64)
    cdef const double[:, :, ::1] b_by_n = b_by_n_arr
    cdef const double[:, :, ::1] core_by_n_T = core_by_n_T_arr
    cdef const double[:, ::1] Phi = Phi_arr
    cdef double[:, ::1] tmp = tmp_arr
    cdef double[:, ::1] out = out_arr
    cdef int ni
    with nogil:
        for ni in range(ndim):
            cy_dgemm(b_by_n[ni, :, :], Phi, tmp)
            cy_dgemm(tmp, core_by_n_T[ni, :, :], out, 1.0, 1.0)
    return out_arr


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
cpdef cnp.ndarray[double, ndim=2] phi_fwd_rhs(
        cnp.ndarray[double, ndim=2] Phi_now,
        cnp.ndarray[double, ndim=3] b_core,
        cnp.ndarray[double, ndim=3] core
):
    cdef int bdim = b_core.shape[0]
    cdef int ndim = b_core.shape[1]
    cdef int Bdim = b_core.shape[2]
    cdef int rdim = core.shape[0]
    cdef int Rdim = core.shape[2]
    cdef cnp.ndarray[double, ndim=3] b_by_n_T_arr = np.ascontiguousarray(np.transpose(b_core, (1, 2, 0)))
    cdef cnp.ndarray[double, ndim=3] core_by_n_arr = np.ascontiguousarray(np.transpose(core, (1, 0, 2)))
    cdef cnp.ndarray[double, ndim=2] Phi_arr = np.ascontiguousarray(Phi_now)
    cdef cnp.ndarray[double, ndim=2] tmp_arr = np.empty((Bdim, rdim), dtype=np.float64)
    cdef cnp.ndarray[double, ndim=2] out_arr = np.zeros((Bdim, Rdim), dtype=np.float64)
    cdef const double[:, :, ::1] b_by_n_T = b_by_n_T_arr
    cdef const double[:, :, ::1] core_by_n = core_by_n_arr
    cdef const double[:, ::1] Phi = Phi_arr
    cdef double[:, ::1] tmp = tmp_arr
    cdef double[:, ::1] out = out_arr
    cdef int ni
    with nogil:
        for ni in range(ndim):
            cy_dgemm(b_by_n_T[ni, :, :], Phi, tmp)
            cy_dgemm(tmp, core_by_n[ni, :, :], out, 1.0, 1.0)
    return out_arr


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
cpdef cnp.ndarray[double, ndim=3] phi_bck_A(
        cnp.ndarray[double, ndim=3] Phi_now,
        cnp.ndarray[double, ndim=3] core_left,
        cnp.ndarray[double, ndim=4] core_A,
        cnp.ndarray[double, ndim=3] core_right
):
    cdef int ldim = core_left.shape[0]
    cdef int Mdim = core_left.shape[1]
    cdef int Ldim = core_left.shape[2]
    cdef int sdim = core_A.shape[0]
    cdef int Ndim = core_A.shape[2]
    cdef int Sdim = core_A.shape[3]
    cdef int rdim = core_right.shape[0]
    cdef int Rdim = core_right.shape[2]
    cdef cnp.ndarray[double, ndim=3] left_by_M_arr = np.ascontiguousarray(np.transpose(core_left, (1, 0, 2)))
    cdef cnp.ndarray[double, ndim=3] Phi_by_S_arr = np.ascontiguousarray(np.transpose(Phi_now, (1, 0, 2)))
    cdef cnp.ndarray[double, ndim=3] right_by_N_T_arr = np.ascontiguousarray(np.transpose(core_right, (1, 2, 0)))
    cdef cnp.ndarray[double, ndim=4] A_by_MNS_arr = np.ascontiguousarray(np.transpose(core_A, (1, 2, 3, 0)))
    cdef cnp.ndarray[double, ndim=2] tmp_arr = np.empty((ldim, Rdim), dtype=np.float64)
    cdef cnp.ndarray[double, ndim=2] small_arr = np.empty((ldim, rdim), dtype=np.float64)
    cdef cnp.ndarray[double, ndim=3] out_arr = np.zeros((ldim, sdim, rdim), dtype=np.float64)
    cdef const double[:, :, ::1] left_by_M = left_by_M_arr
    cdef const double[:, :, ::1] Phi_by_S = Phi_by_S_arr
    cdef const double[:, :, ::1] right_by_N_T = right_by_N_T_arr
    cdef const double[:, :, :, ::1] A_by_MNS = A_by_MNS_arr
    cdef double[:, ::1] tmp = tmp_arr
    cdef double[:, ::1] small = small_arr
    cdef double[:, :, ::1] out = out_arr
    cdef int Mi, Ni, Si, si, li, ri
    cdef double coeff
    with nogil:
        for Mi in range(Mdim):
            for Si in range(Sdim):
                cy_dgemm(left_by_M[Mi, :, :], Phi_by_S[Si, :, :], tmp)
                for Ni in range(Ndim):
                    cy_dgemm(tmp, right_by_N_T[Ni, :, :], small)
                    for si in range(sdim):
                        coeff = A_by_MNS[Mi, Ni, Si, si]
                        if coeff != 0.0:
                            for li in range(ldim):
                                for ri in range(rdim):
                                    out[li, si, ri] += coeff * small[li, ri]
    return out_arr


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
cpdef cnp.ndarray[double, ndim=3] phi_fwd_A(
        cnp.ndarray[double, ndim=3] Phi_now,
        cnp.ndarray[double, ndim=3] core_left,
        cnp.ndarray[double, ndim=4] core_A,
        cnp.ndarray[double, ndim=3] core_right
):
    cdef int ldim = core_left.shape[0]
    cdef int Mdim = core_left.shape[1]
    cdef int Ldim = core_left.shape[2]
    cdef int sdim = core_A.shape[0]
    cdef int Ndim = core_A.shape[2]
    cdef int Sdim = core_A.shape[3]
    cdef int rdim = core_right.shape[0]
    cdef int Rdim = core_right.shape[2]
    cdef cnp.ndarray[double, ndim=3] left_by_M_T_arr = np.ascontiguousarray(np.transpose(core_left, (1, 2, 0)))
    cdef cnp.ndarray[double, ndim=3] Phi_by_s_arr = np.ascontiguousarray(np.transpose(Phi_now, (1, 0, 2)))
    cdef cnp.ndarray[double, ndim=3] right_by_N_arr = np.ascontiguousarray(np.transpose(core_right, (1, 0, 2)))
    cdef cnp.ndarray[double, ndim=4] A_arr = np.ascontiguousarray(core_A)
    cdef cnp.ndarray[double, ndim=2] tmp_arr = np.empty((Ldim, rdim), dtype=np.float64)
    cdef cnp.ndarray[double, ndim=2] small_arr = np.empty((Ldim, Rdim), dtype=np.float64)
    cdef cnp.ndarray[double, ndim=3] out_arr = np.zeros((Ldim, Sdim, Rdim), dtype=np.float64)
    cdef const double[:, :, ::1] left_by_M_T = left_by_M_T_arr
    cdef const double[:, :, ::1] Phi_by_s = Phi_by_s_arr
    cdef const double[:, :, ::1] right_by_N = right_by_N_arr
    cdef const double[:, :, :, ::1] A = A_arr
    cdef double[:, ::1] tmp = tmp_arr
    cdef double[:, ::1] small = small_arr
    cdef double[:, :, ::1] out = out_arr
    cdef int si, Mi, Ni, Si, Li, Ri
    cdef double coeff
    with nogil:
        for si in range(sdim):
            for Mi in range(Mdim):
                cy_dgemm(left_by_M_T[Mi, :, :], Phi_by_s[si, :, :], tmp)
                for Ni in range(Ndim):
                    cy_dgemm(tmp, right_by_N[Ni, :, :], small)
                    for Si in range(Sdim):
                        coeff = A[si, Mi, Ni, Si]
                        if coeff != 0.0:
                            for Li in range(Ldim):
                                for Ri in range(Rdim):
                                    out[Li, Si, Ri] += coeff * small[Li, Ri]
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
