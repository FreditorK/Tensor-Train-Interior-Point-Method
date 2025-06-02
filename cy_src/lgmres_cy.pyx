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
from scipy.sparse.linalg._isolve.iterative import _get_atol_rtol
from numpy.linalg import LinAlgError
from scipy.linalg import (get_blas_funcs, qr_insert)
from scipy.linalg.lapack import get_lapack_funcs, _compute_lwork

cnp.import_array() # Initialize NumPy C-API

@cython.boundscheck(False)
@cython.wraparound(False)
cdef tuple _fgmres(object matvec, cnp.ndarray[double, ndim=2] v0, int m, double atol, cnp.ndarray[double, ndim=2] outer_v, int outer_len):
    cdef:
        object axpy, dot, scal, nrm2
        int i = 0
        int j = 0
        bint b_breakdown = False
        double w_norm
        double res = np.nan
        double eps = np.finfo(v0.dtype).eps
        double alpha = 0.0
        cnp.ndarray[double, ndim = 1] hcur
        cnp.ndarray[double, ndim = 2] Q, R, Q2, R2, y, w, z, zs, vs, a1, b1

    axpy, dot, scal, nrm2 = get_blas_funcs(['axpy', 'dot', 'scal', 'nrm2'], (v0,))

    m = m + outer_len
    vs = np.zeros((v0.shape[0], m + 1), dtype=v0.dtype)
    zs = np.zeros((v0.shape[0], m), dtype=v0.dtype)
    vs[:, 0] = v0.flatten()

    Q = np.ones((1, 1), dtype=v0.dtype)
    R = np.zeros((1, 0), dtype=v0.dtype)

    for j in range(m):
        if j >= m - outer_len:
            z = outer_v[:, j - (m - outer_len)].reshape(-1,  1)
        else:
            z = vs[:, j].reshape(-1,  1)

        w = matvec(z)
        w_norm = nrm2(w)

        hcur = np.zeros(j+2, dtype=Q.dtype)
        for i in range(j+1):
            alpha = dot(vs[:, i], w)
            hcur[i] = alpha
            w = axpy(vs[:, i], w, vs[:, i].shape[0], -alpha)
        hcur[i+1] = nrm2(w)

        with np.errstate(over='ignore', divide='ignore'):
            alpha = 1 / hcur[j+1]

        if np.isfinite(alpha):
            w = scal(alpha, w)

        if not (hcur[j+1] > eps * w_norm):
            b_breakdown = True

        vs[:, j + 1] = w.flatten()
        zs[:, j] = z.flatten()

        Q2 = np.zeros((j+2, j+2), dtype=Q.dtype, order='F')
        Q2[:j+1,:j+1] = Q
        Q2[j+1,j+1] = 1

        R2 = np.zeros((j+2, j), dtype=R.dtype, order='F')
        R2[:j+1,:] = R

        Q, R = qr_insert(Q2, R2, hcur, j, which='col', overwrite_qru=True, check_finite=False)

        res = abs(Q[0, Q.shape[1] - 1])

        if res < atol or b_breakdown:
            break

    if not np.isfinite(R[j,j]):
        raise LinAlgError()

    a1 = R[:j+1,:j+1]
    b1 = Q[0,:j+1].T.reshape(-1, 1)
    lapack_func, lapack_lwork = get_lapack_funcs(('gelsd','gelsd_lwork'), (a1, b1))
    lwork, iwork = _compute_lwork(lapack_lwork, j+1, j+1, j+1, eps)
    y, _, _, info = lapack_func(a1, b1, lwork, iwork, eps, False, False)
    if info != 0:
        raise LinAlgError(f"LAPACK gelsd failed with info={info}")

    return zs, y.flatten(), res

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef lgmres(object A, cnp.ndarray[double, ndim=2] b, double rtol=1e-5, double atol=0., int maxiter=1000, int inner_m=30, int outer_k=3):

    cdef:
        object matvec
        double b_norm, r_norm, inner_res_0, ptol, ptol_max_factor = 1.0
        int k_outer, nx
        int outer_idx = 0
        double pres  # Preconditioned residual norm
        cnp.ndarray[double, ndim=1] x, dx
        cnp.ndarray[double, ndim=2] r_outer, v0
        cnp.ndarray[double, ndim = 2] zs  # List of NumPy arrays representing basis vectors
        cnp.ndarray[double, ndim = 1] y  # Coefficients for basis vectors
        cnp.ndarray[double, ndim = 2] outer_v = np.zeros((b.shape[0], outer_k), dtype=np.float64)

    if not np.isfinite(b).all():
        raise ValueError("RHS must contain only finite numbers")

    matvec = A.matvec  # Python callable, kept as is

    x = np.zeros(b.shape[0])
    gemv, dot, scal, nrm2 = get_blas_funcs(['gemv', 'dot', 'scal', 'nrm2'], (b,))
    b_norm = nrm2(b)
    atol, rtol = _get_atol_rtol('lgmres', b_norm, atol, rtol)

    if b_norm == 0:
        return b, 0

    for k_outer in range(maxiter):
        r_outer = matvec(x.reshape(-1, 1)) - b

        r_norm = np.linalg.norm(r_outer)
        if r_norm <= max(atol, rtol * b_norm):
            break

        v0 = -r_outer
        inner_res_0 = np.linalg.norm(v0)

        if inner_res_0 == 0:
            raise RuntimeError(f"Returned a zero vector; |v| ~ {r_norm:.1g}, |M v| = 0")

        v0 = scal(1.0 / inner_res_0, v0)
        ptol = min(ptol_max_factor, max(atol, rtol * b_norm) / r_norm)

        try:
            zs, y, pres = _fgmres(
                matvec, v0,
                inner_m,
                ptol,
                outer_v,
                min(outer_idx,  outer_k)
            )
            y = scal(inner_res_0, y)
            if not np.isfinite(y).all():
                raise LinAlgError()
        except LinAlgError:
            return x, k_outer + 1

        if pres > ptol:
            ptol_max_factor = min(1.0, 1.5 * ptol_max_factor)
        else:
            ptol_max_factor = max(1e-16, 0.25 * ptol_max_factor)

        dx = gemv(1.0, zs[:, :len(y)], y, beta=0.0)
        nx = np.linalg.norm(dx)
        if nx > 0:
            outer_v[:, outer_idx % outer_k] = dx / nx
            outer_idx += 1

        x += dx
    else:
        return x, maxiter

    return x, 0
