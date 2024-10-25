# tt_ops_cy.pyx
# cython: language_level=3

cdef extern from "numpy/arrayobject.h":
    # Define NPY_NO_DEPRECATED_API for compatibility with numpy
    ctypedef void npy_no_deprecated_api

import numpy as np
cimport numpy as cnp  # This allows Cython to understand NumPy's C-API



cpdef list tt_identity(int dim):
    cdef list result = []
    cdef cnp.ndarray[cnp.float64_t, ndim=4] I = np.eye(2).reshape(1, 2, 2, 1)
    cdef int i

    for i in range(dim):
        result.append(I)

    return result

cpdef list tt_zero_matrix(int dim, tuple shape=(2, 2)):
    cdef list result = []
    cdef cnp.ndarray[cnp.float64_t, ndim=4] zeros_array = np.zeros((1, *shape, 1))
    cdef int i

    for i in range(dim):
        result.append(zeros_array)

    return result
 

cpdef list tt_one_matrix(int dim, tuple shape=(2, 2)):
    cdef list result = []
    cdef cnp.ndarray[cnp.float64_t, ndim=4] ones_array = np.ones((1, *shape, 1))
    cdef int i

    for i in range(dim):
        result.append(ones_array)

    return result

cpdef list tt_transpose(list matrix_tt):
    cdef list shape_lengths = [len(c.shape) for c in matrix_tt]
    cdef int split_idx = np.argmax(shape_lengths)
    cdef list transposed_cores = matrix_tt[:split_idx]
    cdef cnp.ndarray[cnp.float64_t, ndim=4] swapped_core
    cdef int iters = len(matrix_tt)
    cdef int i

    for i in range(split_idx, iters):
        swapped_core = matrix_tt[i]
        swapped_core = np.swapaxes(swapped_core, axis1=1, axis2=2)
        transposed_cores.append(swapped_core)

    return transposed_cores


cpdef list tt_ranks(list train_tt):
    cdef int n = len(train_tt)
    cdef list ranks = []
    cdef int rank
    cdef cnp.ndarray core
    cdef int i

    for i in range(1, n):
        core = train_tt[i]
        rank = len(core)
        ranks.append(rank)

    return ranks


cpdef list tt_scale(float alpha, list train_tt):
    cdef int n = len(train_tt)
    cdef int idx = np.random.randint(low=0, high=n)
    cdef list scaled_tt = []
    cdef cnp.ndarray core
    cdef int i

    for i in range(idx):
        core = train_tt[i]
        scaled_tt.append(core)

    core = alpha * train_tt[idx]
    scaled_tt.append(core)

    for i in range(idx + 1, n):
        core = train_tt[i]
        scaled_tt.append(core)

    return scaled_tt

cpdef list tt_swap_all(list tt_train):
    cdef int n = len(tt_train)
    cdef list swapped_tt = []
    cdef int i
    cdef cnp.ndarray core

    for i in range(n - 1, -1, -1):
        core = tt_train[i]
        core = np.swapaxes(core, 0, -1)
        swapped_tt.append(core)

    return swapped_tt