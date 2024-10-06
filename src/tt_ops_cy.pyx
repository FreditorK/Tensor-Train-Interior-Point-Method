# tt_ops_cy.pyx

cdef extern from "numpy/arrayobject.h":
    # Define NPY_NO_DEPRECATED_API for compatibility with numpy
    ctypedef void npy_no_deprecated_api

import numpy as np
cimport numpy as np  # This allows Cython to understand NumPy's C-API
from typing import List
from itertools import product



def tt_identity(int dim) -> List[np.ndarray]:
    cdef list result = []
    cdef np.ndarray[np.float64_t, ndim=4] I = np.array([[1.0, 0], [0, 1.0]]).reshape(1, 2, 2, 1)

    for _ in range(dim):
        result.append(I)

    return result

def tt_zero_matrix(int dim, tuple shape=(2, 2)) -> List[np.ndarray]:
    cdef list result = []
    cdef np.ndarray[np.float64_t, ndim=4] zeros_array = np.zeros((1, *shape, 1))

    for _ in range(dim):
        result.append(zeros_array)

    return result
 

def tt_one_matrix(int dim, tuple shape=(2, 2)) -> List[np.ndarray]:
    cdef list result = []
    cdef np.ndarray[np.float64_t, ndim=4] ones_array = np.ones((1, *shape, 1))

    for _ in range(dim):
        result.append(ones_array)

    return result

def tt_transpose(List[np.ndarray] matrix_tt) -> List[np.ndarray]:
    cdef int split_idx = np.argmax([len(c.shape) for c in matrix_tt])
    cdef list transposed_cores = matrix_tt[:split_idx]
    transposed_cores.extend(np.swapaxes(c, axis1=1, axis2=2) for c in matrix_tt[split_idx:])

    return transposed_cores


def tt_adjoint(List[np.ndarray] linear_op_tt) -> List[np.ndarray]:
    cdef int n = len(linear_op_tt)
    cdef np.ndarray core
    cdef list adjoint_cores = []
    for i in range(n):
        core = linear_op_tt[i]
        core = np.swapaxes(core, axis1=2, axis2=3)
        adjoint_cores.append(core)

    return adjoint_cores


def tt_ranks(List[np.ndarray] train_tt) -> List[int]:
    cdef int n = len(train_tt)
    cdef list ranks = []
    cdef int rank 

    for i in range(n - 1):
        rank = train_tt[i].shape[-1]
        ranks.append(rank)

    return ranks


def tt_scale(float alpha, List[np.ndarray] train_tt) -> List[np.ndarray]:
    cdef int idx = np.random.randint(low=0, high=len(train_tt))
    cdef list scaled_tt = []

    for i in range(idx):
        scaled_tt.append(train_tt[i])

    scaled_tt.append(alpha * train_tt[idx])

    for i in range(idx + 1, len(train_tt)):
        scaled_tt.append(train_tt[i])

    return scaled_tt

def tt_swap_all(List[np.ndarray] tt_train) -> List[np.ndarray]:
    cdef int n = len(tt_train)
    cdef list swapped_tt = [] 
    cdef int i

    for i in range(n - 1, -1, -1): 
        swapped_tt.append(np.swapaxes(tt_train[i], 0, -1))

    return swapped_tt