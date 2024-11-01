# ops_cy.pyx
# cython: language_level=3

cdef extern from "numpy/arrayobject.h":
    # Define NPY_NO_DEPRECATED_API for compatibility with numpy
    ctypedef void npy_no_deprecated_api

import numpy as np
cimport numpy as cnp  # This allows Cython to understand NumPy's C-API
import scipy as scip

cpdef cnp.ndarray[double, ndim=2] compute_phi_fwd_rhs(cnp.ndarray[double, ndim=2] Phi_now,
                                                     cnp.ndarray[double, ndim=3] core_rhs,
                                                     cnp.ndarray[double, ndim=3] core):

    cdef cnp.ndarray[double, ndim=2] Phi_next = np.einsum('br,bnB,rnR->BR', Phi_now, core_rhs, core, optimize=True)
    return Phi_next

cpdef cnp.ndarray[double, ndim=2] compute_phi_bck_rhs(cnp.ndarray[double, ndim=2] Phi_now,
                                                     cnp.ndarray[double, ndim=3] core_b,
                                                     cnp.ndarray[double, ndim=3] core):

    cdef cnp.ndarray[double, ndim=2] Phi = np.einsum('BR,bnB,rnR->br', Phi_now, core_b, core, optimize=True)

    return Phi

cpdef cnp.ndarray[double, ndim=3] compute_phi_fwd_A(cnp.ndarray[double, ndim=3] Phi_now,
                                                   cnp.ndarray[double, ndim=3] core_left,
                                                   cnp.ndarray[double, ndim=4] core_A,
                                                   cnp.ndarray[double, ndim=3] core_right):

    cdef cnp.ndarray[double, ndim=3] Phi_next = np.einsum('lsr,lML,sMNS,rNR->LSR', Phi_now, core_left, core_A, core_right, optimize=True)

    return Phi_next

cpdef cnp.ndarray[double, ndim=3] compute_phi_bck_A(cnp.ndarray[double, ndim=3] Phi_now,
                                                   cnp.ndarray[double, ndim=3] core_left,
                                                   cnp.ndarray[double, ndim=4] core_A,
                                                   cnp.ndarray[double, ndim=3] core_right):

    cdef cnp.ndarray[double, ndim=3] Phi = np.einsum('LSR,lML,sMNS,rNR->lsr', Phi_now, core_left, core_A, core_right, optimize=True)

    return Phi

cpdef cnp.ndarray[double, ndim=3] local_product(cnp.ndarray[double, ndim=3] Phi_right,
                                               cnp.ndarray[double, ndim=3] Phi_left,
                                               cnp.ndarray[double, ndim=4] coreA,
                                               cnp.ndarray[double, ndim=3] core):

    cdef cnp.ndarray[double, ndim=3] w = np.einsum('lsr,smnS,LSR,rnR->lmL', Phi_left, coreA, Phi_right, core, optimize=True)

    return w

cpdef tuple compute_phi_bcks_rhs(list Phis_b, list b, list x_cores, int d):

    cdef cnp.ndarray[double, ndim=2] current_Phis_b
    cdef cnp.ndarray[double, ndim=2] next_Phis_b
    cdef cnp.ndarray[double, ndim=3] current_b
    cdef cnp.ndarray[double, ndim=3] x_core
    cdef cnp.ndarray[double, ndim=1] normb = np.ones(d - 1)
    cdef int k
    cdef tuple return_tuple

    for k in range(d - 1, 0, -1):
        current_Phis_b = Phis_b[k]
        next_Phis_b = Phis_b[k+1]
        current_b = b[k]
        x_core = x_cores[k]
        current_Phis_b = compute_phi_bck_rhs(next_Phis_b, current_b, x_core)

        norm = np.linalg.norm(current_Phis_b)
        norm = norm if norm > 0 else 1.0

        normb[k - 1] = norm
        Phis_b[k] = current_Phis_b / norm

    return_tuple = (Phis_b, normb)
    return return_tuple


cpdef tuple compute_phi_bcks_A(list Phis, list z_cores, list matrix_tt, list x_cores, int d):
    cdef double norm
    cdef cnp.ndarray[double, ndim=3] current_phi
    cdef cnp.ndarray[double, ndim=3] next_phi
    cdef cnp.ndarray[double, ndim=3] x_core
    cdef cnp.ndarray[double, ndim=3] z_core
    cdef cnp.ndarray[double, ndim=4] matrix_core
    cdef cnp.ndarray[double, ndim=1] normA = np.ones(d-1)
    cdef int k
    cdef tuple return_tuple

    for k in range(d - 1, 0, -1):
        current_phi = Phis[k]
        next_phi = Phis[k + 1]
        x_core = x_cores[k]
        z_core = z_cores[k]
        matrix_core = matrix_tt[k]

        current_phi = compute_phi_bck_A(next_phi, z_core, matrix_core, x_core)

        norm = np.linalg.norm(current_phi)
        norm = norm if np.greater(norm, 0) else 1.0
        current_phi = np.divide(current_phi, norm)

        normA[k-1] = norm
        Phis[k] = current_phi

    return_tuple = (Phis, normA)

    return return_tuple

cpdef tuple solution_truncation(cnp.ndarray[double, ndim=2] solution_now,
                                cnp.ndarray[double, ndim=3] current_phi,
                                cnp.ndarray[double, ndim=3] next_phi,
                                cnp.ndarray[double, ndim=4] current_matrix_core,
                                cnp.ndarray[double, ndim=2] rhs,
                                int rank,
                                int current_N,
                                int rankp1,
                                double tol):
    cdef tuple solution_tuple
    cdef cnp.ndarray[double, ndim=2] U
    cdef cnp.ndarray[double, ndim=1] s
    cdef cnp.ndarray[double, ndim=2] S
    cdef cnp.ndarray[double, ndim=2] V
    cdef cnp.ndarray[double, ndim=3] local_core
    cdef cnp.ndarray[double, ndim=2] trunc_solution
    cdef int r = 0
    cdef int iter
    cdef double res

    U, s, V = scip.linalg.svd(solution_now, full_matrices=False)
    iter = U.shape[1]
    for r in range(iter - 1, 0, -1):
        S = np.diag(s[:r])
        trunc_solution = U[:, :r] @ S @ V[:r, :]
        solution = trunc_solution.reshape((rank, current_N, rankp1))
        local_core = np.einsum('lsr,smnS,LSR,rnR->lmL', current_phi, current_matrix_core, next_phi, solution, optimize=True)
        res = np.linalg.norm(local_core.reshape(rhs.shape[0], rhs.shape[1]) - rhs)

        if res > tol:
            break

    r = min(r + 1, len(s))
    U = U[:, :r]
    V = np.diag(s[:r]) @ V[:r, :]
    solution_tuple = (U, V.T)
    return solution_tuple

