# ops_cy.pyx
# cython: language_level=3

cdef extern from "numpy/arrayobject.h":
    # Define NPY_NO_DEPRECATED_API for compatibility with numpy
    ctypedef void npy_no_deprecated_api

import numpy as np
cimport numpy as np  # This allows Cython to understand NumPy's C-API
from typing import List

cpdef np.ndarray[double, ndim=2] compute_phi_fwd_rhs(np.ndarray[double, ndim=2] Phi_now,
                                                     np.ndarray[double, ndim=3] core_rhs,
                                                     np.ndarray[double, ndim=3] core):

    cdef np.ndarray[double, ndim=2] Phi_next = np.einsum('br,bnB,rnR->BR', Phi_now, core_rhs, core)
    return Phi_next

cpdef np.ndarray[double, ndim=2] _compute_phi_bck_rhs(np.ndarray[double, ndim=2] Phi_now,
                                                      np.ndarray[double, ndim=3] core_b,
                                                      np.ndarray[double, ndim=3] core):

    cdef np.ndarray[double, ndim=2] Phi = np.einsum('BR,bnB,rnR->br', Phi_now, core_b, core)

    return Phi

cpdef np.ndarray[double, ndim=3] compute_phi_fwd_A(np.ndarray[double, ndim=3] Phi_now,
                                                   np.ndarray[double, ndim=3] core_left,
                                                   np.ndarray[double, ndim=4] core_A,
                                                   np.ndarray[double, ndim=3] core_right):

    cdef np.ndarray[double, ndim=3] Phi_next = np.einsum('lsr,lML,sMNS,rNR->LSR', Phi_now, core_left, core_A, core_right)

    return Phi_next

cpdef np.ndarray[double, ndim=3] _compute_phi_bck_A(np.ndarray[double, ndim=3] Phi_now,
                                                    np.ndarray[double, ndim=3] core_left,
                                                    np.ndarray[double, ndim=4] core_A,
                                                    np.ndarray[double, ndim=3] core_right):

    cdef np.ndarray[double, ndim=3] Phi = np.einsum('LSR,lML,sMNS,rNR->lsr', Phi_now, core_left, core_A, core_right)

    return Phi

cpdef np.ndarray[double, ndim=3] local_product(np.ndarray[double, ndim=3] Phi_right,
                                               np.ndarray[double, ndim=3] Phi_left,
                                               np.ndarray[double, ndim=4] coreA,
                                               np.ndarray[double, ndim=3] core,
                                               int bandA=-1):

    cdef np.ndarray[double, ndim=3] w
    cdef np.ndarray[double, ndim=3] tmp
    cdef int i
    cdef int zero = 0

    if bandA < 0:
        w = np.einsum('lsr,smnS,LSR,rnR->lmL', Phi_left, coreA, Phi_right, core)
    else:
        # Initialize w with zeros
        w = np.zeros((Phi_left.shape[0], coreA.shape[1], Phi_right.shape[0]), dtype=np.float64)

        for i in range(-bandA, bandA + 1):
            # Extract diagonal and pad
            tmp = np.diagonal(coreA, i, axis1=1, axis2=2)  # Extract the diagonal
            tmp = np.pad(tmp, ((i, zero) if i > zero else (zero, abs(i)),
                               (abs(i), zero) if i < zero else (zero, i),
                               (zero, zero)))  # Pad the diagonal appropriately

            # Calculate tmp with einsum
            tmp = np.einsum('lsr,sSm,LSR,rmR->lmL', Phi_left, tmp, Phi_right, core)

            # Pad tmp based on i
            if i < 0:
                tmp = np.pad(tmp[:, :i, :], ((0, 0), (0, 0), (-i, 0)))
            else:
                tmp = np.pad(tmp[:, i:, :], ((0, 0), (0, 0), (0, i)))

            # Accumulate the results
            w += tmp

    return w

cpdef tuple compute_phi_bcks_rhs(list Phis_b, list b, list x_cores, int d):

    cdef np.ndarray[double, ndim=2] current_Phis_b
    cdef np.ndarray[double, ndim=2] next_Phis_b
    cdef np.ndarray[double, ndim=3] current_b
    cdef np.ndarray[double, ndim=3] x_core
    cdef np.ndarray[double, ndim=1] normb = np.ones(d - 1)
    cdef tuple return_tuple

    for k in range(d - 1, 0, -1):
        current_Phis_b = Phis_b[k]
        next_Phis_b = Phis_b[k+1]
        current_b = b[k]
        x_core = x_cores[k]
        current_Phis_b = _compute_phi_bck_rhs(next_Phis_b, current_b, x_core)

        norm = np.linalg.norm(current_Phis_b)
        norm = norm if norm > 0 else 1.0

        normb[k - 1] = norm
        Phis_b[k] = current_Phis_b / norm

    return_tuple = (Phis_b, normb)
    return return_tuple


cpdef tuple compute_phi_bcks_A(list Phis, list z_cores, list matrix_tt, list x_cores, int d):
    cdef int k
    cdef double norm
    cdef np.ndarray[double, ndim=3] current_phi
    cdef np.ndarray[double, ndim=3] next_phi
    cdef np.ndarray[double, ndim=3] x_core
    cdef np.ndarray[double, ndim=3] z_core
    cdef np.ndarray[double, ndim=4] matrix_core
    cdef np.ndarray[double, ndim=1] normA = np.ones(d-1)
    cdef tuple return_tuple

    for k in range(d - 1, 0, -1):
        current_phi = Phis[k]
        next_phi = Phis[k + 1]
        x_core = x_cores[k]
        z_core = z_cores[k]
        matrix_core = matrix_tt[k]

        current_phi = _compute_phi_bck_A(next_phi, z_core, matrix_core, x_core)

        norm = np.linalg.norm(current_phi)
        norm = norm if np.greater(norm, 0) else 1.0
        current_phi = np.divide(current_phi, norm)

        normA[k-1] = norm
        Phis[k] = current_phi

    return_tuple = (Phis, normA)

    return return_tuple

cpdef tuple solution_truncation(np.ndarray[double, ndim=2] solution_now,
                                np.ndarray[double, ndim=3] current_phi,
                                np.ndarray[double, ndim=3] next_phi,
                                np.ndarray[double, ndim=4] current_matrix_core,
                                np.ndarray[double, ndim=2] rhs,
                                int rank,
                                int current_N,
                                int rankp1,
                                double tol):
    cdef tuple solution_tuple
    cdef np.ndarray[double, ndim=2] U
    cdef np.ndarray[double, ndim=1] s
    cdef np.ndarray[double, ndim=2] S
    cdef np.ndarray[double, ndim=2] V
    cdef int r = 0

    U, s, V = np.linalg.svd(solution_now, full_matrices=False)
    for r in range(U.shape[1] - 1, 0, -1):
        S = np.diag(s[:r])
        solution = U[:, :r] @ S @ V[:r, :]
        solution = np.reshape(solution, (rank, current_N, rankp1))

        res = np.linalg.norm(np.reshape(local_product(next_phi, current_phi, current_matrix_core, solution),(-1, 1)) - rhs)

        if res > tol:
            break
    r = min(r + 1, len(s))
    U = U[:, :r]
    V = np.diag(s[:r]) @ V[:r, :]
    solution_tuple = (U, V.T)
    return solution_tuple

