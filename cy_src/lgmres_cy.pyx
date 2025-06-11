# tt_ops_cy.pyx
# cython: language_level=3
# cython: cdivision=True
# cython: optimize.use_switch=True

cdef extern from "numpy/arrayobject.h":
    # Define NPY_NO_DEPRECATED_API for compatibility with numpy
    ctypedef void npy_no_deprecated_api

import numpy as np
cimport numpy as cnp
cimport cython
cimport scipy.linalg.cython_blas as blas
from numpy.linalg import LinAlgError
from scipy.linalg import qr_insert
from scipy.linalg.cython_lapack cimport dgelsd, dtrtrs
from libc.stdlib cimport malloc, free

cnp.import_array() # Initialize NumPy C-API


cdef:
    int inc = 1  # typical unit stride
    double global_alpha = 1.0

@cython.boundscheck(False)
@cython.wraparound(False)
cdef void cy_axpy(double[:] dx, double[:] x):
    cdef int n = len(x)
    blas.daxpy(&n, &global_alpha, &dx[0], &inc, &x[0], &inc)

@cython.boundscheck(False)
@cython.wraparound(False)
cdef void cy_full_axpy(double[:] dx, double[:] x, double alpha):
    cdef int n = len(x)
    blas.daxpy(&n, &alpha, &dx[0], &inc, &x[0], &inc)

@cython.boundscheck(False)
@cython.wraparound(False)
cdef double cy_dot(double[:] x, double[:] y) nogil:
    cdef int n = len(x)
    return blas.ddot(&n, &x[0], &inc, &y[0], &inc)

@cython.boundscheck(False)
@cython.wraparound(False)
cdef void cy_scal(double alpha, double[:] x):
    cdef int n = len(x)
    blas.dscal(&n, &alpha, &x[0], &inc)

@cython.boundscheck(False)
@cython.wraparound(False)
cdef double cy_nrm2(double[:] x) nogil:
    cdef int n = len(x)
    return blas.dnrm2(&n, &x[0], &inc)

@cython.boundscheck(False)
@cython.wraparound(False)
cdef double[:] cy_solve_upper_triangular(double[:, :] a, double[:] b):
    # --- C-level variable declarations ---
    cdef int n = a.shape[0]
    cdef int nrhs = 1
    cdef int lda = max(1, n)
    cdef int ldb = max(1, n)
    cdef int info = 0

    # --- Hard-coded LAPACK parameters ---
    cdef char uplo = b'U'
    cdef char trans = b'N'
    cdef char diag = b'N'

    # --- Input validation ---
    if a.shape[0] != a.shape[1]:
        raise ValueError("Matrix 'a' must be square.")
    if a.shape[0] != b.shape[0]:
        raise ValueError("Matrices 'a' and 'b' must have the same number of rows.")

    with nogil:
        dtrtrs(&uplo, &trans, &diag,
               &n, &nrhs,
               &a[0, 0], &lda,
               &b[0], &ldb,
               &info)

    # --- Error handling ---
    if info < 0:
        raise ValueError(f"dtrtrs: illegal value in internal argument {-info}")
    if info > 0:
        raise np.linalg.LinAlgError(
            f"dtrtrs: the {info}-th diagonal element of A is zero, matrix is singular"
        )

    return b


cdef class BaseMatVec:

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef cnp.ndarray[double, ndim=3] einsum(self, double[:, :, :] XAX, double[:, :, :, :] block_A, double[:, :, :] XAX1, double[:, :, :] x_core):
        # lsr,smnS,LSR,rnR->lmL
        cdef double[:, :, :, :] intermediate_1 = np.tensordot(x_core, XAX1, axes=([2], [2]))
        cdef double[:, :, :, :] intermediate_2 = np.tensordot(intermediate_1, block_A, axes=([1, 3], [2, 3]))
        cdef double[:, :, :] result = np.tensordot(intermediate_2, XAX, axes=([0, 2], [2, 1]))
        return np.transpose(result, axes=(2, 1, 0))

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef cnp.ndarray[double, ndim=1] matvec(self, cnp.ndarray[double, ndim=1] x_core):
        raise NotImplementedError("BaseMatVec.matvec must be implemented by subclass")

cdef class MatVecWrapper(BaseMatVec):
    cdef double[:,  :,  :] XAX_k_00, XAX_k_01, XAX_k_01T, XAX_k_21, XAX_k_22
    cdef double[:,  :,  :, :] block_A_k_00, block_A_k_01, block_A_k_01T, block_A_k_21, block_A_k_22
    cdef double[:,  :,  :] XAX_kp1_00, XAX_kp1_01, XAX_kp1_01T, XAX_kp1_21, XAX_kp1_22
    cdef double[:,  :,  :] inv_I
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
                 int x0,
                 int x2,
                 int x3):
        self.XAX_k_00 = XAX_k_00
        self.XAX_k_01 = XAX_k_01
        self.XAX_k_01T = np.transpose(XAX_k_01,  axes=(2, 1, 0))
        self.XAX_k_21 = XAX_k_21
        self.XAX_k_22 = XAX_k_22

        self.block_A_k_00 = block_A_k_00
        self.block_A_k_01 = block_A_k_01
        self.block_A_k_01T = np.transpose(block_A_k_01, axes=(0, 2, 1, 3))
        self.block_A_k_21 = block_A_k_21
        self.block_A_k_22 = block_A_k_22

        self.XAX_kp1_00 = XAX_kp1_00
        self.XAX_kp1_01 = XAX_kp1_01
        self.XAX_kp1_01T = np.transpose(XAX_kp1_01,  axes=(2, 1, 0))
        self.XAX_kp1_21 = XAX_kp1_21
        self.XAX_kp1_22 = XAX_kp1_22

        self.inv_I = inv_I
        self.r = x0
        self.n = x2
        self.R = x3

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef cnp.ndarray[double, ndim=1] matvec(self, cnp.ndarray[double, ndim=1] x_core):
        cdef double[:, :,  :, :] x_reshaped = x_core.view().reshape(2, self.r, self.n, self.R)
        cdef cnp.ndarray[double, ndim=4] result = np.empty_like(x_reshaped, dtype=np.float64)

        result[0] = self.einsum(self.XAX_k_00, self.block_A_k_00, self.XAX_kp1_00, x_reshaped[0])
        result[1] = self.einsum(self.XAX_k_21, self.block_A_k_21, self.XAX_kp1_21, x_reshaped[1])
        result[0] += self.einsum(self.XAX_k_01, self.block_A_k_01, self.XAX_kp1_01, x_reshaped[1])
        cdef cnp.ndarray[double, ndim=3] temp = self.einsum(self.XAX_k_01T, self.block_A_k_01T, self.XAX_kp1_01T, x_reshaped[0])
        temp *= self.inv_I
        result[1] -= self.einsum(self.XAX_k_22, self.block_A_k_22, self.XAX_kp1_22, temp)
        return result.ravel()

cdef class IneqMatVecWrapper(BaseMatVec):
    cdef double[:,  :,  :] XAX_k_00, XAX_k_01, XAX_k_01T, XAX_k_21, XAX_k_22, XAX_k_31,XAX_k_33
    cdef double[:,  :,  :, :] block_A_k_00, block_A_k_01, block_A_k_01T, block_A_k_21, block_A_k_22, block_A_k_31, block_A_k_33
    cdef double[:,  :,  :] XAX_kp1_00, XAX_kp1_01, XAX_kp1_01T, XAX_kp1_21, XAX_kp1_22, XAX_kp1_31, XAX_kp1_33
    cdef double[:, :, :] inv_I
    cdef int r, n, R

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
                 int x0,
                 int x2,
                 int x3):
        self.XAX_k_00 = XAX_k_00
        self.XAX_k_01 = XAX_k_01
        self.XAX_k_01T = np.transpose(XAX_k_01, axes=(2, 1, 0))
        self.XAX_k_21 = XAX_k_21
        self.XAX_k_22 = XAX_k_22
        self.XAX_k_31 = XAX_k_31
        self.XAX_k_33 = XAX_k_33

        self.block_A_k_00 = block_A_k_00
        self.block_A_k_01 = block_A_k_01
        self.block_A_k_01T = np.transpose(block_A_k_01, axes=(0, 2, 1, 3))
        self.block_A_k_21 = block_A_k_21
        self.block_A_k_22 = block_A_k_22
        self.block_A_k_31 = block_A_k_31
        self.block_A_k_33 = block_A_k_33

        self.XAX_kp1_00 = XAX_kp1_00
        self.XAX_kp1_01 = XAX_kp1_01
        self.XAX_kp1_01T = np.transpose(XAX_kp1_01, axes=(2, 1, 0))
        self.XAX_kp1_21 = XAX_kp1_21
        self.XAX_kp1_22 = XAX_kp1_22
        self.XAX_kp1_31 = XAX_kp1_31
        self.XAX_kp1_33 = XAX_kp1_33

        self.inv_I = inv_I
        self.r = x0
        self.n = x2
        self.R = x3

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef cnp.ndarray[double, ndim=1] matvec(self, cnp.ndarray[double, ndim=1] x_core):
        cdef cnp.ndarray[double, ndim=4] x = x_core.reshape((3, self.r, self.n, self.R))
        cdef cnp.ndarray[double, ndim=4] result = np.zeros_like(x, dtype=np.float64)

        result[0] = self.einsum(self.XAX_k_00, self.block_A_k_00, self.XAX_kp1_00, x[0])
        result[1] = self.einsum(self.XAX_k_21, self.block_A_k_21, self.XAX_kp1_21, x[1])
        result[2] = self.einsum(self.XAX_k_31, self.block_A_k_31, self.XAX_kp1_31, x[1])
        result[0] += self.einsum(self.XAX_k_01, self.block_A_k_01, self.XAX_kp1_01, x[1])
        cdef cnp.ndarray[double, ndim=3] temp = self.einsum(self.XAX_k_01T, self.block_A_k_01T, self.XAX_kp1_01T, x[0])
        temp *= self.inv_I
        temp += x[2]
        result[1] -= self.einsum(self.XAX_k_22, self.block_A_k_22, self.XAX_kp1_22, temp)
        result[2] += self.einsum(self.XAX_k_33, self.block_A_k_33, self.XAX_kp1_33, x[2])

        return result.ravel()


@cython.boundscheck(False)
@cython.wraparound(False)
cdef tuple _fgmres(BaseMatVec linear_op, cnp.ndarray[double, ndim=1] v0, int m, double atol, cnp.ndarray[double, ndim=2] outer_v, int outer_len):
    cdef:
        int i = 0
        int j = 0
        bint b_breakdown = False
        double w_norm
        double res = np.nan
        double eps = np.finfo(v0.dtype).eps
        double alpha = 0.0
        cnp.ndarray[double, ndim = 1] hcur, z, w, y
        cnp.ndarray[double, ndim = 2] Q, R, zs, vs
        cnp.ndarray[double, ndim = 1, mode='fortran'] b1
        cnp.ndarray[double, ndim = 2, mode='fortran'] Q2, R2, a1

    m = m + outer_len
    vs = np.zeros((v0.shape[0], m + 1), dtype=v0.dtype, order='F')
    zs = np.zeros((v0.shape[0], m), dtype=v0.dtype, order='F')
    vs[:, 0] = v0

    Q = np.ones((1, 1), dtype=v0.dtype)
    R = np.zeros((1, 0), dtype=v0.dtype)
    Q2 = np.zeros((m+2, m+2), dtype=Q.dtype, order='F')
    R2 = np.zeros((m + 2, m), dtype=R.dtype, order='F')
    hcur = np.zeros(m+2, dtype=Q.dtype)

    for j in range(m):
        if j >= m - outer_len:
            z = outer_v[:, j - (m - outer_len)]
        else:
            z = vs[:, j]

        w = linear_op.matvec(z)
        w_norm = cy_nrm2(w)

        for i in range(j+1):
            alpha = cy_dot(vs[:, i], w)
            hcur[i] = alpha
            cy_full_axpy(vs[:, i], w, -alpha)
        hcur[i+1] = cy_nrm2(w)

        with np.errstate(over='ignore', divide='ignore'):
            alpha = 1 / hcur[j+1]

        if np.isfinite(alpha):
            cy_scal(alpha, w)

        if not (hcur[j+1] > eps * w_norm):
            b_breakdown = True

        vs[:, j + 1] = w
        zs[:, j] = z

        Q2[:j+1,:j+1] = Q
        Q2[j+1,j+1] = 1

        R2[:j+1,:j] = R

        Q, R = qr_insert(Q2[:j+2, :j+2], R2[:j+2,  :j], hcur[:j+2], j, which='col', overwrite_qru=True, check_finite=False) # this is a cython call

        res = abs(Q[0, Q.shape[1] - 1])

        if res < atol or b_breakdown:
            break

    if not np.isfinite(R[j,j]):
        raise LinAlgError()

    a1 = np.asfortranarray(R[:j+1,:j+1])
    b1 = np.asfortranarray(Q[0,:j+1].ravel())
    y = np.asarray(cy_solve_upper_triangular(a1, b1))

    return zs, y, res

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef tuple lgmres(BaseMatVec linear_op, cnp.ndarray[double, ndim=1] b, double rtol=1e-5, double atol=0., int maxiter=1000, int inner_m=30, int outer_k=3):

    cdef:
        double b_norm, r_norm, inner_res_0, ptol, ptol_max_factor = 1.0
        int k_outer
        int outer_idx = 0
        double pres, nx  # Preconditioned residual norm
        cnp.ndarray[double, ndim=1] x, dx, r_outer, v0, y
        cnp.ndarray[double, ndim = 2] zs  # List of NumPy arrays representing basis vectors
        cnp.ndarray[double, ndim = 2] outer_v = np.zeros((b.shape[0], outer_k), dtype=np.float64)

    x = np.zeros(b.shape[0])
    b_norm = cy_nrm2(b)
    atol = max(atol, rtol * b_norm)

    if b_norm == 0:
        return b, 0

    for k_outer in range(maxiter):
        r_outer = linear_op.matvec(x) - b

        r_norm = cy_nrm2(r_outer)
        if r_norm <= max(atol, rtol * b_norm):
            break

        v0 = -r_outer
        inner_res_0 = cy_nrm2(v0)

        if inner_res_0 == 0:
            raise RuntimeError(f"Returned a zero vector; |v| ~ {r_norm:.1g}, |M v| = 0")

        cy_scal(1.0 / inner_res_0, v0)
        ptol = min(ptol_max_factor, max(atol, rtol * b_norm) / r_norm)

        try:
            zs, y, pres = _fgmres(
                linear_op, v0,
                inner_m,
                ptol,
                outer_v,
                min(outer_idx,  outer_k)
            )
            cy_scal(inner_res_0, y)
            if not np.isfinite(y).all():
                raise LinAlgError()
        except LinAlgError:
            return x, k_outer + 1

        if pres > ptol:
            ptol_max_factor = min(1.0, 1.5 * ptol_max_factor)
        else:
            ptol_max_factor = max(1e-16, 0.25 * ptol_max_factor)

        dx = (zs[:, :len(y)] @ y.reshape(-1, 1)).ravel()
        nx = cy_nrm2(dx)
        if nx > 0:
            outer_v[:, outer_idx % outer_k] = dx / nx
            outer_idx += 1

        cy_axpy(dx, x)
    else:
        return x, maxiter

    return x, 0