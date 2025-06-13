# tt_ops_cy.pyx
# cython: language_level=3
# cython: cdivision=True
# cython: optimize.use_switch=True
# distutils: language = c++

cdef extern from "numpy/arrayobject.h":
    # Define NPY_NO_DEPRECATED_API for compatibility with numpy
    ctypedef void npy_no_deprecated_api

cdef extern from "lapacke.h":
    void dgetrf_(int *m, int *n, double *a, int *lda, int *ipiv, int *info) nogil
    void dgetrs_(char *trans, int *n, int *nrhs, double *a, int *lda,
                int *ipiv, double *b, int *ldb, int *info, int trans_len) nogil

import numpy as np
cimport numpy as cnp  # This allows Cython to understand NumPy's C-API
cimport cython
import scipy as scp

cnp.import_array() # Initialize NumPy C-API


from scipy.linalg.cython_lapack cimport dtrtrs


@cython.boundscheck(False)
@cython.wraparound(False)
cdef cnp.ndarray[double, ndim=2] lu_solve(cnp.ndarray[double, ndim=2] A, cnp.ndarray[double, ndim=2] b):
    cdef cnp.ndarray[double, ndim=2] A_fact
    if A.flags.c_contiguous:
        A_fact = A
    else:
        A_fact = np.ascontiguousarray(A)

    cdef cnp.ndarray[double, ndim=2] x
    if b.flags.f_contiguous:
        x = b
    else:
        x = np.asfortranarray(b)

    cdef int m = A_fact.shape[0]
    cdef int n = A_fact.shape[1]

    cdef int nrhs = 1
    cdef int lda = n
    cdef int ldb = m
    cdef int info = 0
    cdef cnp.ndarray[int, ndim=1] ipiv = np.empty(m, dtype=np.int32)

    with nogil:
        dgetrf_(&m, &n, &A_fact[0, 0], &lda, &ipiv[0], &info)

    if info < 0:
        raise ValueError('dgetrf: illegal value in argument %d' % -info)
    elif info > 0:
        raise ValueError('dgetrf: U(%d, %d) is exactly zero, matrix is singular' % (info, info))

    with nogil:
        dgetrs_(b'T', &n, &nrhs, &A_fact[0, 0], &lda, &ipiv[0], &x[0, 0], &ldb, &info, 1)

    if info != 0:
        # dgetrs does not return positive info values.
        raise ValueError('dgetrs: illegal value in argument %d' % -info)
        
    return x


@cython.boundscheck(False)
@cython.wraparound(False)
cdef cnp.ndarray[double, ndim=2] forward_backward_sub(cnp.ndarray[double, ndim=2] L,
                                  cnp.ndarray[double, ndim=2] b,
                                  bint overwrite_b=False):
    if not L.flags.c_contiguous:
        L = np.ascontiguousarray(L)

    cdef int n = L.shape[0]
    cdef int nrhs = b.shape[1] if b.ndim > 1 else 1
    cdef int lda = n
    cdef int ldb = n
    cdef int info = 0
    cdef cnp.ndarray[double, ndim=2] y

    if overwrite_b:
        y = b
        if not y.flags.f_contiguous:
            y = np.asfortranarray(y)
    else:
        y = np.asfortranarray(b)

    
    with nogil:
        dtrtrs(b'U', b'T', b'N', &n, &nrhs, &L[0, 0], &lda, &y[0, 0], &ldb, &info)

    if info < 0:
        raise ValueError('dtrtrs: illegal value in argument %d' % -info)
    elif info > 0:
        raise ValueError('dtrtrs: matrix is singular in forward pass')
    
    with nogil:
        dtrtrs(b'U', b'N', b'N', &n, &nrhs, &L[0, 0], &lda, &y[0, 0], &ldb, &info)
    
    if info < 0:
        raise ValueError('dtrtrs: illegal value in argument %d' % -info)
    elif info > 0:
        raise ValueError('dtrtrs: matrix is singular in backward pass')

    return y

@cython.boundscheck(False)
@cython.wraparound(False)
cdef cnp.ndarray[double, ndim=5] tensordot1(cnp.ndarray[double, ndim=4] B, cnp.ndarray[double, ndim=3] C):
    cdef int s = B.shape[0]
    cdef int m = B.shape[1]
    cdef int n = B.shape[2]
    cdef int S = B.shape[3]
    cdef int L = C.shape[0]
    cdef int R = C.shape[2]

    cdef cnp.ndarray[double, ndim=2] B_flat = B.reshape(s * m * n, S)
    cdef cnp.ndarray[double, ndim=2] C_flat = C.transpose(1, 0, 2).reshape(S, L * R)
    cdef cnp.ndarray[double, ndim=2] result_flat = B_flat @ C_flat
    return result_flat.reshape(s, m, n, L, R)

@cython.boundscheck(False)
@cython.wraparound(False)
cdef cnp.ndarray[double, ndim=6] tensordot2(cnp.ndarray[double, ndim=3] A, cnp.ndarray[double, ndim=5] temp):
    cdef int l = A.shape[0]
    cdef int s = A.shape[1]
    cdef int r = A.shape[2]
    cdef int m = temp.shape[1]
    cdef int n = temp.shape[2]
    cdef int L = temp.shape[3]
    cdef int R = temp.shape[4]

    cdef cnp.ndarray[double, ndim=2] A_flat = A.transpose(0, 2, 1).reshape(l * r, s)
    cdef cnp.ndarray[double, ndim=2] temp_flat = temp.reshape(s, m * n * L * R)
    cdef cnp.ndarray[double, ndim=2] result_flat = A_flat @ temp_flat
    return result_flat.reshape(l, r, m, n, L, R)

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef cnp.ndarray[double, ndim=2] einsum(
        double[:, :, :] A,
        double[:, :, :, :] B,
        double[:, :, :] C,
        int m):
    # Step 1: Contract B ('smnS') and C ('LSR') over 'S'.
    # The result has indices 'smnLR'.
    cdef cnp.ndarray[double, ndim=5] temp
    temp = tensordot1(np.asarray(B), np.asarray(C))

    # Step 2: Contract A ('lsr') with the intermediate result ('smnLR') over 's'.
    # The result has indices 'lrmnLR'.
    cdef cnp.ndarray[double, ndim=6] res_unordered
    res_unordered = tensordot2(np.asarray(A), temp)

    # Step 3: Transpose the result from 'lrmnLR' to the desired 'lmLrnR' order.
    # Current order: 0, 1, 2, 3, 4, 5 (l, r, m, n, L, R)
    # Target order:  0, 2, 4, 1, 3, 5 (l, m, L, r, n, R)
    cdef cnp.ndarray[double, ndim=6] res_ordered
    res_ordered = np.transpose(res_unordered, axes=(0, 2, 4, 1, 3, 5))
    
    # Step 4: Reshape the final 6D array into a 2D matrix of shape (m, m).
    return res_ordered.reshape(m, m)


cdef class SchurSolver:
    cdef double[:,  :] mR_p, mR_d, mR_c
    cdef double[:,  :, :] XAX_k_00, XAX_k_01, XAX_k_21, XAX_k_22
    cdef double[:,  :, :, :] block_A_k_00, block_A_k_01, block_A_k_01T, block_A_k_21, block_A_k_22
    cdef double[:,  :, :] XAX_kp1_00, XAX_kp1_01, XAX_kp1_21, XAX_kp1_22
    cdef double[:,  :, :] inv_I
    cdef int r, n, R  # shape dims

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
                 cnp.ndarray[double, ndim=2] mR_p,
                 cnp.ndarray[double, ndim=2] mR_d,
                 cnp.ndarray[double, ndim=2] mR_c,
                 int r,
                 int n,
                 int R):
        self.XAX_k_00 = XAX_k_00
        self.XAX_k_01 = XAX_k_01
        self.XAX_k_21 = XAX_k_21
        self.XAX_k_22 = XAX_k_22

        self.block_A_k_00 = block_A_k_00
        self.block_A_k_01 = block_A_k_01
        self.block_A_k_21 = block_A_k_21
        self.block_A_k_22 = block_A_k_22

        self.XAX_kp1_00 = XAX_kp1_00
        self.XAX_kp1_01 = XAX_kp1_01
        self.XAX_kp1_21 = XAX_kp1_21
        self.XAX_kp1_22 = XAX_kp1_22

        self.r = r
        self.n = n
        self.R = R
        self.inv_I = inv_I

        self.mR_p = mR_p
        self.mR_d = mR_d
        self.mR_c = mR_c

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef cnp.ndarray[double, ndim=4] solve(self, cnp.ndarray[double, ndim=4] previous_solution):
        cdef cnp.ndarray[double, ndim=2] L_L_Z, L_X, mL_eq, A, b, L_L_Z_input
        cdef cnp.ndarray[double, ndim=2] mR_p, mR_d, mR_c, temp_solve, b_temp_solve
        cdef cnp.ndarray[double, ndim=4] solution_now = np.empty((self.r, 3, self.n, self.R))
        cdef int m = self.r * self.n * self.R

        L_L_Z_input = einsum(self.XAX_k_21, self.block_A_k_21, self.XAX_kp1_21, m)
        L_L_Z = scp.linalg.cholesky(L_L_Z_input, check_finite=False, lower=True, overwrite_a=True)
        L_X = einsum(self.XAX_k_22, self.block_A_k_22, self.XAX_kp1_22, m)
        mL_eq = einsum(self.XAX_k_01, self.block_A_k_01, self.XAX_kp1_01, m)

        inv_I_np = np.asarray(self.inv_I)

        temp_solve = forward_backward_sub(L_L_Z, L_X * inv_I_np.reshape(1, -1), overwrite_b=True)
        A = (mL_eq @ temp_solve @ mL_eq.T)
        A += einsum(self.XAX_k_00, self.block_A_k_00, self.XAX_kp1_00, m)

        b_temp_solve = forward_backward_sub(L_L_Z, self.mR_c - (L_X * inv_I_np.reshape(1, -1)) @ self.mR_d, overwrite_b=True)
        b = self.mR_p - mL_eq @ b_temp_solve - A @ previous_solution[:, 0].reshape(-1, 1)
        
        # Solve for the first part of the solution
        sol_0 = lu_solve(A, b).reshape(self.r, self.n, self.R)
        solution_now[:, 0] = sol_0 + previous_solution[:, 0]
        
        # Calculate the second part of the solution
        sol_2_flat = (self.mR_d - mL_eq.T @ solution_now[:, 0].reshape(-1, 1)) * inv_I_np.reshape(-1, 1)
        solution_now[:, 2] = sol_2_flat.reshape(self.r, self.n, self.R)
        
        # Calculate the third part of the solution
        sol_1_input = self.mR_c 
        sol_1_input -= L_X @ solution_now[:, 2].reshape(-1, 1)
        solution_now[:, 1] = forward_backward_sub(L_L_Z, sol_1_input, overwrite_b=True).reshape(self.r, self.n, self.R)

        return solution_now