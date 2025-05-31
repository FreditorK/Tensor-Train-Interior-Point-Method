# tt_ops_cy.pyx
# cython: language_level=3

cdef extern from "numpy/arrayobject.h":
    # Define NPY_NO_DEPRECATED_API for compatibility with numpy
    ctypedef void npy_no_deprecated_api

import numpy as np
cimport numpy as cnp  # This allows Cython to understand NumPy's C-API
cimport cython
import scipy as scp


# cython: boundscheck=False, wraparound=False, cdivision=True

import numpy as np
cimport numpy as cnp
from scipy.linalg import get_blas_funcs
from scipy.sparse.linalg._isolve.iterative import _get_atol_rtol
from scipy.sparse.linalg._isolve._gcrotmk import _fgmres
from collections import deque
from numpy.linalg import LinAlgError

cpdef lgmres(object A, cnp.ndarray[double, ndim=2] b, double rtol=1e-5, double atol=0.,
           int maxiter=1000, int inner_m=30, int outer_k=3):

    cdef:
        cnp.ndarray[double, ndim=2] x, r_outer, v0, dx
        double b_norm, r_norm, inner_res_0, ptol, ptol_max_factor = 1.0
        object matvec, dot = None, scal = None, nrm2 = None
        object outer_v = deque(maxlen=outer_k)
        int k_outer, nx
        object Q, R, B, vs, zs, y, pres

    if not np.isfinite(b).all():
        raise ValueError("RHS must contain only finite numbers")

    matvec = A.matvec  # Python callable, kept as is

    x = np.zeros_like(b)
    nrm2 = get_blas_funcs('nrm2', [b])
    b_norm = nrm2(b)

    atol, rtol = _get_atol_rtol('lgmres', b_norm, atol, rtol)

    if b_norm == 0:
        return b, 0

    for k_outer in range(maxiter):
        r_outer = matvec(x) - b

        if np.iscomplexobj(r_outer) and not np.iscomplexobj(x):
            x = x.astype(r_outer.dtype)
        dot, scal, nrm2 = get_blas_funcs(['dot', 'scal', 'nrm2'], (x, r_outer))

        r_norm = nrm2(r_outer)
        if r_norm <= max(atol, rtol * b_norm):
            break

        v0 = -r_outer
        inner_res_0 = nrm2(v0)

        if inner_res_0 == 0:
            raise RuntimeError(f"Returned a zero vector; |v| ~ {r_norm:.1g}, |M v| = 0")

        v0 = scal(1.0 / inner_res_0, v0)
        ptol = min(ptol_max_factor, max(atol, rtol * b_norm) / r_norm)

        try:
            Q, R, B, vs, zs, y, pres = _fgmres(
                matvec, v0,
                inner_m,
                atol=ptol,
                outer_v=outer_v,
                prepend_outer_v=False
            )
            y *= inner_res_0
            if not np.isfinite(y).all():
                raise LinAlgError()
        except LinAlgError:
            return x, k_outer + 1

        if pres > ptol:
            ptol_max_factor = min(1.0, 1.5 * ptol_max_factor)
        else:
            ptol_max_factor = max(1e-16, 0.25 * ptol_max_factor)

        dx = y[0] * zs[0] + np.tensordot(y[1:], zs[1:], axes=1)
        nx = nrm2(dx)
        if nx > 0:
            outer_v.append((dx / nx, None))

        x += dx
    else:
        return x, maxiter

    return x, 0
