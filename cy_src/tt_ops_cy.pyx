# tt_ops_cy.pyx
# cython: language_level=3
# cython: cdivision=True
# cython: optimize.use_switch=True
# distutils: language = c++

cdef extern from "numpy/arrayobject.h":
    # Define NPY_NO_DEPRECATED_API for compatibility with numpy
    ctypedef void npy_no_deprecated_api

import numpy as np
cimport numpy as cnp  # This allows Cython to understand NumPy's C-API
cimport cython
from scipy.linalg.cython_blas cimport dgemm
import scipy as scp
from opt_einsum import contract as einsum

cnp.import_array() # Initialize NumPy C-API

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.inline
cdef void cy_dgemm_row(
        const double[:, ::1] A,
        const double[:, ::1] B,
        double[:, ::1] C,
        double alpha=1.0,
        double beta=0.0
) noexcept nogil:
    cdef int M = A.shape[0]
    cdef int K = A.shape[1]
    cdef int N = B.shape[1]
    cdef char trans = 78
    dgemm(&trans, &trans, &N, &M, &K, &alpha,
          <double*>&B[0, 0], &N, <double*>&A[0, 0], &K, &beta, &C[0, 0], &N)

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef list tt_identity(int dim):
    cdef list result = [None] * dim  # Preallocate the list
    cdef cnp.ndarray[double, ndim=4] I = np.eye(2).reshape(1, 2, 2, 1)
    cdef int i

    for i in range(dim):
        result[i] = I

    return result

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef list tt_zero_matrix(int dim):
    cdef list result = [None] * dim  # Preallocate the list
    cdef cnp.ndarray[double, ndim=4] zeros_array = np.zeros((1, 2, 2, 1))
    cdef int i

    for i in range(dim):
        result[i] = zeros_array

    return result
 
@cython.boundscheck(False)
@cython.wraparound(False)
cpdef list tt_one_matrix(int dim):
    cdef list result = [None] * dim
    cdef cnp.ndarray[double, ndim=4] ones_array = np.ones((1, 2, 2, 1))
    cdef int i

    for i in range(dim):
        result[i] = ones_array

    return result

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef list tt_transpose(list matrix_tt):
    cdef Py_ssize_t split_idx = 0
    cdef Py_ssize_t iters = len(matrix_tt)
    cdef Py_ssize_t i
    cdef cnp.ndarray[double, ndim=4] core
    cdef cnp.ndarray[double, ndim=4] swapped_core
    cdef list transposed_cores = [None] * iters

    # Determine split index based on the maximum shape length
    split_idx = np.argmax([np.ndim(c) for c in matrix_tt])

    # Copy first part without change
    for i in range(split_idx):
        transposed_cores[i] = matrix_tt[i]

    # Transpose from split_idx onward
    for i in range(split_idx, iters):
        core = matrix_tt[i]
        swapped_core = np.swapaxes(core, 1, 2)
        transposed_cores[i] = swapped_core

    return transposed_cores

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef list tt_ranks(list train_tt):
    cdef int n = len(train_tt)
    cdef int i
    cdef cnp.ndarray core
    cdef list ranks = [0] * (n - 1)  # Preallocate result list

    for i in range(1, n):
        core = train_tt[i]
        ranks[i - 1] = core.shape[0]  # Or len(core) if it's guaranteed to be 1D along axis 0

    return ranks

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef list tt_scale(float alpha, list train_tt):
    cdef int n = len(train_tt)
    cdef int i, idx = 0
    cdef list scaled_tt = [None] * n
    cdef cnp.ndarray core

    # Copy unchanged cores before the scaled one
    for i in range(idx):
        scaled_tt[i] = train_tt[i]

    # Scale selected core
    core = train_tt[idx]
    scaled_tt[idx] = alpha * core

    # Copy remaining cores after the scaled one
    for i in range(idx + 1, n):
        scaled_tt[i] = train_tt[i]

    return scaled_tt

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef list tt_swap_all(list tt_train):
    cdef int n = len(tt_train)
    cdef list swapped_tt = [None] * n  # preallocate list
    cdef int i
    cdef cnp.ndarray core

    for i in range(n):
        core = tt_train[n - 1 - i]
        swapped_tt[i] = np.swapaxes(core, 0, -1)

    return swapped_tt

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef tt_rl_orthogonalise(list train_tt):
    cdef int dim = len(train_tt)
    cdef int i
    cdef tuple shape_i, shape_im1
    cdef int shape_length
    cdef int new_rank
    cdef cnp.ndarray q_T, r
    if dim == 1:
        return train_tt
    for i in range(dim - 1, 0, -1):
        shape_i = train_tt[i].shape
        shape_im1 = train_tt[i - 1].shape
        shape_length = len(shape_i)

        # QR decomposition
        q_T, r = scp.linalg.qr(
            train_tt[i].reshape(shape_i[0], np.prod(shape_i[1:])).T,
            check_finite=False,
            mode="economic"
        )
        new_rank = r.shape[0]

        train_tt[i] = q_T.T.reshape(new_rank, *shape_i[1:])
        train_tt[i - 1] = (
                train_tt[i - 1].reshape(np.prod(shape_im1[:shape_length-1]), shape_i[0]) @ r.T
        ).reshape(*shape_im1[:shape_length-1], new_rank)

    return train_tt

@cython.boundscheck(False)
cpdef int prune_singular_vals(cnp.ndarray[cnp.double_t, ndim=1] s, double eps):
    cdef double norm_s = np.linalg.norm(s)
    cdef cnp.ndarray[cnp.double_t, ndim=1] sc
    cdef int R

    if norm_s == 0.0:
        return 1

    sc = np.cumsum(np.abs(s[::-1]) ** 2)[::-1]

    R = np.argmax(sc < eps ** 2)
    R = max(R, 1)
    if sc[-1] > eps ** 2:
        R = s.size

    return R

@cython.boundscheck(False)
cpdef list tt_rank_reduce(list train_tt, double eps=1e-18):
    cdef int dim = len(train_tt)
    cdef list ranks_py = tt_ranks(train_tt)
    cdef cnp.ndarray[int, ndim=1] ranks = np.array([1] + ranks_py + [1], dtype=np.int32)

    if dim == 1 or np.all(ranks == 1):
        return train_tt

    eps = eps / np.sqrt(dim - 1)
    train_tt = tt_rl_orthogonalise(train_tt)

    cdef int rank = 1
    cdef int idx, next_rank
    cdef tuple idx_shape, next_idx_shape
    cdef cnp.ndarray u, s, v_t
    cdef cnp.ndarray reshaped_core, reshaped_next

    for idx in range(dim - 1):
        idx_shape = train_tt[idx].shape
        next_idx_shape = train_tt[idx + 1].shape

        reshaped_core = train_tt[idx].reshape(
            rank * int(np.prod(idx_shape[1:-1], dtype=np.int32)), -1
        )

        u, s, v_t = scp.linalg.svd(
            reshaped_core,
            full_matrices=False,
            check_finite=False,
            overwrite_a=True,
            lapack_driver="gesvd"
        )

        next_rank = prune_singular_vals(s, eps)

        train_tt[idx] = u[:, :next_rank].reshape(
            rank, *idx_shape[1:-1], next_rank
        )

        reshaped_next = train_tt[idx + 1].reshape(next_idx_shape[0], -1)
        train_tt[idx + 1] = (
            s[:next_rank].reshape(-1, 1) * v_t[:next_rank, :] @ reshaped_next
        ).reshape(next_rank, *next_idx_shape[1:-1], -1)

        rank = next_rank

    return train_tt

@cython.boundscheck(False)
cpdef cnp.ndarray _block_diag_tensor(object tensor_1, object tensor_2):
    """
    For internal use: Concatenates two tensors to a block diagonal tensor.
    Works for tensors with shape (r1, n1, ..., nd, r2)
    """

    cdef tuple shape_1 = tensor_1.shape
    cdef tuple shape_2 = tensor_2.shape
    cdef cnp.ndarray result = np.zeros((shape_1[0] + shape_2[0], *shape_1[1:-1], shape_1[-1] + shape_2[-1]))
    N = tuple([slice(s) for s in shape_1[1:-1]])
    result[(slice(0, shape_1[0]), *N, slice(0, shape_1[-1]))] = tensor_1
    result[(slice(shape_1[0], shape_1[0] + shape_2[0]), *N, slice(shape_1[-1], shape_1[-1] + shape_2[-1]))] = tensor_2
    return result

@cython.boundscheck(False)
cpdef tt_add(list train_1_tt, list train_2_tt):
    """
    Adds two tensor trains
    """
    cdef int n = len(train_1_tt)
    if n > 1:
        return [
            np.concatenate((train_1_tt[0], train_2_tt[0]), axis=-1)
        ] + [
            _block_diag_tensor(core_1, core_2) for core_1, core_2 in zip(train_1_tt[1:-1], train_2_tt[1:-1])
        ] + [
            np.concatenate((train_1_tt[-1], train_2_tt[-1]), axis=0)
        ]
    else:
        return [train_1_tt[0] + train_2_tt[0]]


@cython.boundscheck(False)
cpdef list tt_psd_rank_reduce(list train_tt, double eps=1e-18):
    cdef int dim = len(train_tt)
    eps /= 2.0

    cdef cnp.ndarray[int, ndim=1] ranks = np.array([1] + tt_ranks(train_tt) + [1], dtype=np.int32)
    if dim == 1 or np.all(ranks == 1):
        return train_tt

    eps = eps / np.sqrt(dim - 1)
    train_tt = tt_rl_orthogonalise(train_tt)

    cdef int rank = 1
    cdef double sum_eps_sq = 0.0
    cdef int idx, next_rank, s_len
    cdef tuple idx_shape, next_idx_shape
    cdef cnp.ndarray u, s, v_t
    cdef cnp.ndarray sc
    cdef double factor
    cdef cnp.ndarray I
    cdef cnp.ndarray reshaped_core, reshaped_next

    for idx in range(dim - 1):
        idx_shape = train_tt[idx].shape
        next_idx_shape = train_tt[idx + 1].shape

        reshaped_core = train_tt[idx].reshape(
            rank * int(np.prod(idx_shape[1:-1], dtype=np.int32)), -1
        )

        u, s, v_t = scp.linalg.svd(
            reshaped_core,
            full_matrices=False,
            check_finite=False,
            overwrite_a=True,
            lapack_driver="gesvd"
        )

        # Squared singular values in descending order
        sc = np.cumsum(np.abs(s[::-1]) ** 2)[::-1]
        s_len = s.shape[0]

        next_rank = np.argmax(sc < eps ** 2)
        next_rank = max(next_rank, 1)
        if sc[-1] > eps ** 2:
            next_rank = s_len

        if next_rank < s_len:
            sum_eps_sq += sc[next_rank]

        train_tt[idx] = u[:, :next_rank].reshape(rank, *idx_shape[1:-1], next_rank)

        reshaped_next = train_tt[idx + 1].reshape(next_idx_shape[0], -1)
        train_tt[idx + 1] = (
                s[:next_rank].reshape(-1, 1) * v_t[:next_rank, :] @ reshaped_next
        ).reshape(next_rank, *next_idx_shape[1:-1], -1)

        rank = next_rank

    factor = pow(sum_eps_sq, 1.0 / (2 * dim))
    I = factor * np.eye(train_tt[0].shape[1]).reshape(
        1, *train_tt[0].shape[1:-1], 1
    )

    return tt_add(train_tt, [I] * dim)


@cython.boundscheck(False)
cpdef list tt_mask_rank_reduce(list train_tt, list mask_tt, double eps=1e-18):
    cdef int dim = len(train_tt)
    eps /= 2.0

    cdef cnp.ndarray[int, ndim=1] ranks = np.array([1] + tt_ranks(train_tt) + [1], dtype=np.int32)
    if dim == 1 or np.all(ranks == 1):
        return train_tt

    eps = eps / np.sqrt(dim - 1)
    train_tt = tt_rl_orthogonalise(train_tt)

    cdef int rank = 1
    cdef double sum_eps_sq = 0.0
    cdef int idx, next_rank, s_len
    cdef tuple idx_shape, next_idx_shape
    cdef cnp.ndarray u, s, v_t
    cdef cnp.ndarray sc
    cdef double factor
    cdef cnp.ndarray reshaped_core, reshaped_next

    for idx in range(dim - 1):
        idx_shape = train_tt[idx].shape
        next_idx_shape = train_tt[idx + 1].shape

        reshaped_core = train_tt[idx].reshape(
            rank * int(np.prod(idx_shape[1:-1], dtype=np.int32)), -1
        )

        u, s, v_t = scp.linalg.svd(
            reshaped_core,
            full_matrices=False,
            check_finite=False,
            overwrite_a=True,
            lapack_driver="gesvd"
        )

        # Squared singular values in descending order
        sc = np.cumsum(np.abs(s[::-1]) ** 2)[::-1]
        s_len = s.shape[0]

        next_rank = np.argmax(sc < eps ** 2)
        next_rank = max(next_rank, 1)
        if sc[-1] > eps ** 2:
            next_rank = s_len

        if next_rank < s_len:
            sum_eps_sq += sc[next_rank]

        train_tt[idx] = u[:, :next_rank].reshape(rank, *idx_shape[1:-1], next_rank)

        reshaped_next = train_tt[idx + 1].reshape(next_idx_shape[0], -1)
        train_tt[idx + 1] = (
                s[:next_rank].reshape(-1, 1) * v_t[:next_rank, :] @ reshaped_next
        ).reshape(next_rank, *next_idx_shape[1:-1], -1)

        rank = next_rank

    factor = pow(sum_eps_sq, 1.0 / (2 * dim))

    return tt_add(train_tt, [factor*c for c in mask_tt])


@cython.boundscheck(False)
@cython.wraparound(False)
cdef cnp.ndarray[double, ndim=2] swap_matrix_3d(
        cnp.ndarray[double, ndim=3] core_a,
        cnp.ndarray[double, ndim=3] core_b
):
    cdef int ra = core_a.shape[0]
    cdef int na = core_a.shape[1]
    cdef int ka = core_a.shape[2]
    cdef int nb = core_b.shape[1]
    cdef int rb = core_b.shape[2]
    cdef cnp.ndarray[double, ndim=3] a_arr = np.ascontiguousarray(core_a)
    cdef cnp.ndarray[double, ndim=2] b_flat_arr = np.ascontiguousarray(core_b.reshape(ka, nb * rb))
    cdef cnp.ndarray[double, ndim=2] work_arr = np.empty((na, nb * rb), dtype=np.float64)
    cdef cnp.ndarray[double, ndim=2] matrix_arr = np.empty((ra * nb, na * rb), dtype=np.float64)
    cdef const double[:, :, ::1] a = a_arr
    cdef const double[:, ::1] b_flat = b_flat_arr
    cdef double[:, ::1] work = work_arr
    cdef double[:, ::1] matrix = matrix_arr
    cdef int ai, ni, bi, ri
    with nogil:
        for ai in range(ra):
            cy_dgemm_row(a[ai, :, :], b_flat, work)
            for bi in range(nb):
                for ni in range(na):
                    for ri in range(rb):
                        matrix[ai * nb + bi, ni * rb + ri] = work[ni, bi * rb + ri]
    return matrix_arr


@cython.boundscheck(False)
@cython.wraparound(False)
cdef cnp.ndarray[double, ndim=2] swap_matrix_4d(
        cnp.ndarray[double, ndim=4] core_a,
        cnp.ndarray[double, ndim=4] core_b
):
    cdef int ra = core_a.shape[0]
    cdef int ma = core_a.shape[1]
    cdef int na = core_a.shape[2]
    cdef int ka = core_a.shape[3]
    cdef int mb = core_b.shape[1]
    cdef int nb = core_b.shape[2]
    cdef int rb = core_b.shape[3]
    cdef cnp.ndarray[double, ndim=3] a_flat_arr = np.ascontiguousarray(core_a.reshape(ra, ma * na, ka))
    cdef cnp.ndarray[double, ndim=2] b_flat_arr = np.ascontiguousarray(core_b.reshape(ka, mb * nb * rb))
    cdef cnp.ndarray[double, ndim=2] work_arr = np.empty((ma * na, mb * nb * rb), dtype=np.float64)
    cdef cnp.ndarray[double, ndim=2] matrix_arr = np.empty((ra * mb * nb, ma * na * rb), dtype=np.float64)
    cdef const double[:, :, ::1] a_flat = a_flat_arr
    cdef const double[:, ::1] b_flat = b_flat_arr
    cdef double[:, ::1] work = work_arr
    cdef double[:, ::1] matrix = matrix_arr
    cdef int ai, mi, ni, bi, bj, ri
    with nogil:
        for ai in range(ra):
            cy_dgemm_row(a_flat[ai, :, :], b_flat, work)
            for bi in range(mb):
                for bj in range(nb):
                    for mi in range(ma):
                        for ni in range(na):
                            for ri in range(rb):
                                matrix[(ai * mb + bi) * nb + bj, (mi * na + ni) * rb + ri] = work[mi * na + ni, (bi * nb + bj) * rb + ri]
    return matrix_arr


@cython.boundscheck(False)
@cython.wraparound(False)
cdef tuple swap_cores(cnp.ndarray core_a, cnp.ndarray core_b, double eps):
    cdef cnp.ndarray reshaped_matrix
    cdef cnp.ndarray u, s, v, core_a_new, core_b_new
    cdef int r_pruned

    if core_a.ndim == 3:
        reshaped_matrix = swap_matrix_3d(core_a, core_b)

        u, s, v = scp.linalg.svd(reshaped_matrix, full_matrices=False, check_finite=False, overwrite_a=True, lapack_driver="gesvd")
        r_pruned = prune_singular_vals(s, eps)

        core_a_new = np.reshape(u[:, :r_pruned] * s[:r_pruned].reshape(1, -1),
                                (core_a.shape[0], core_b.shape[1], -1))
        core_b_new = np.reshape(v[:r_pruned, :],
                                (-1, core_a.shape[1], core_b.shape[2]))

        return core_a_new, core_b_new
    reshaped_matrix = swap_matrix_4d(core_a, core_b)

    u, s, v = scp.linalg.svd(reshaped_matrix, full_matrices=False, check_finite=False, overwrite_a=True, lapack_driver="gesvd")
    r_pruned = prune_singular_vals(s, eps)

    core_a_new = np.reshape(u[:, :r_pruned] * s[:r_pruned].reshape(1, -1),
                            (core_a.shape[0], core_b.shape[1], core_b.shape[2], -1))
    core_b_new = np.reshape(v[:r_pruned, :],
                            (-1, core_a.shape[1], core_a.shape[2], core_b.shape[3]))

    return core_a_new, core_b_new

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef list tt_fast_matrix_vec_mul(list matrix_tt, list vec_tt, double eps=1e-18):
    """
    Cython implementation of fast matrix-vector multiplication for Tensor Trains.
    Based on the algorithm described in https://arxiv.org/pdf/2410.19747
    """
    cdef int dim = len(matrix_tt)
    cdef double loop_eps = eps / np.sqrt(dim - 1) if dim > 1 else eps
    cdef list cores = [np.transpose(c, (2, 1, 0)) for c in reversed(vec_tt)]

    cdef int i, j
    for i in range(dim):
        cores[0] = np.tensordot(matrix_tt[dim - i - 1], cores[0], axes=([3, 2], [0, 1]))

        if i != dim - 1:
            for j in range(i, -1, -1):
                cores[j], cores[j + 1] = swap_cores(cores[j], cores[j + 1], loop_eps)

    return cores

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef list tt_fast_mat_mat_mul(list matrix_tt_1, list matrix_tt_2, double eps=1e-18):
    cdef int dim = len(matrix_tt_1)
    cdef double loop_eps = eps / np.sqrt(dim - 1) if dim > 1 else eps
    cdef list cores = [np.transpose(c, (3, 1, 2, 0)) for c in reversed(matrix_tt_2)]

    cdef int i, j
    for i in range(dim):
        cores[0] = np.tensordot(matrix_tt_1[dim - i - 1], cores[0], axes=([3, 2], [0, 1]))

        if i != dim - 1:
            for j in range(i, -1, -1):
                cores[j], cores[j + 1] = swap_cores(cores[j], cores[j + 1], loop_eps)

    return cores

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef list tt_fast_hadamard(list train_tt_1, list train_tt_2, double eps=1e-18):
    cdef int dim = len(train_tt_1)
    cdef double loop_eps = eps / np.sqrt(dim - 1) if dim > 1 else eps
    cdef list cores
    cdef int i, j
    cdef cnp.ndarray current_core_1, current_core_2, tensor_contraction, diag_contraction
    if len(train_tt_1[0].shape) == 4 and len(train_tt_2[0].shape) == 4:
        cores = [np.transpose(c, (3, 1, 2, 0)) for c in reversed(train_tt_2)]
        for i in range(dim):
            current_core_1 = train_tt_1[dim - i - 1]
            current_core_2 = cores[0]
            tensor_contraction = np.tensordot(current_core_1, current_core_2, axes=([3], [0]))
            diag_contraction = np.diagonal(tensor_contraction, axis1=1, axis2=3)
            diag_contraction = np.diagonal(diag_contraction, axis1=1, axis2=2)
            cores[0] = diag_contraction.transpose(0, 2, 3, 1)

            if i != dim - 1:
                for j in range(i, -1, -1):
                    cores[j], cores[j + 1] = swap_cores(cores[j], cores[j + 1], loop_eps)

        return cores
    else:
        cores = [np.transpose(c, (2, 1, 0)) for c in reversed(train_tt_2)]
        for i in range(dim):
            current_core_1 = train_tt_1[dim - i - 1]
            current_core_2 = cores[0]
            tensor_contraction = np.tensordot(current_core_1, current_core_2, axes=([2],[0]))
            diag_contraction = np.diagonal(tensor_contraction, axis1=1, axis2=2)
            cores[0] = diag_contraction.transpose(0, 2, 1)

            if i != dim - 1:
                for j in range(i, -1, -1):
                    cores[j], cores[j + 1] = swap_cores(cores[j], cores[j + 1], loop_eps)

        return cores

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
cdef cnp.ndarray[double, ndim=2] inner_step(
        cnp.ndarray[double, ndim=2] result,
        cnp.ndarray core1,
        cnp.ndarray core2
):
    cdef int left1 = core1.shape[0]
    cdef int right1 = core1.shape[core1.ndim - 1]
    cdef int left2 = core2.shape[0]
    cdef int right2 = core2.shape[core2.ndim - 1]
    cdef int phys = 1
    cdef int i, j, p
    for i in range(1, core1.ndim - 1):
        phys *= core1.shape[i]

    cdef cnp.ndarray[double, ndim=2] result_T_arr = np.ascontiguousarray(result.T)
    cdef cnp.ndarray[double, ndim=2] core1_flat_arr = np.ascontiguousarray(core1.reshape(left1, phys * right1))
    cdef cnp.ndarray[double, ndim=3] core2_phys_arr = np.ascontiguousarray(core2.reshape(left2, phys, right2))
    cdef cnp.ndarray[double, ndim=2] tmp_arr = np.empty((left2, phys * right1), dtype=np.float64)
    cdef cnp.ndarray[double, ndim=3] tmp_phys_arr = tmp_arr.reshape(left2, phys, right1)
    cdef cnp.ndarray[double, ndim=2] tmp_slice_arr = np.empty((right1, left2), dtype=np.float64)
    cdef cnp.ndarray[double, ndim=2] core2_slice_arr = np.empty((left2, right2), dtype=np.float64)
    cdef cnp.ndarray[double, ndim=2] out_arr = np.zeros((right1, right2), dtype=np.float64)
    cdef const double[:, ::1] result_T = result_T_arr
    cdef const double[:, ::1] core1_flat = core1_flat_arr
    cdef const double[:, :, ::1] core2_phys = core2_phys_arr
    cdef const double[:, :, ::1] tmp_phys = tmp_phys_arr
    cdef double[:, ::1] tmp = tmp_arr
    cdef double[:, ::1] tmp_slice = tmp_slice_arr
    cdef double[:, ::1] core2_slice = core2_slice_arr
    cdef double[:, ::1] out = out_arr
    with nogil:
        cy_dgemm_row(result_T, core1_flat, tmp)
        for p in range(phys):
            for i in range(right1):
                for j in range(left2):
                    tmp_slice[i, j] = tmp_phys[j, p, i]
            for i in range(left2):
                for j in range(right2):
                    core2_slice[i, j] = core2_phys[i, p, j]
            cy_dgemm_row(tmp_slice, core2_slice, out, 1.0, 1.0)
    return out_arr


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef double tt_inner_prod(list train_1_tt, list train_2_tt):
    cdef cnp.ndarray[double, ndim=2] result = np.array([[1.0]], dtype=np.float64)
    cdef cnp.ndarray core1, core2
    cdef tuple core_pair
    for core_pair in zip(train_1_tt, train_2_tt):
        core1, core2 = core_pair
        result = inner_step(result, core1, core2)

    return result[0, 0]

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef list tt_normalise(list train_tt, int radius=1):
    cdef double factor = np.divide(radius, np.sqrt(tt_inner_prod(train_tt, train_tt)))
    return tt_scale(factor, train_tt)

@cython.boundscheck(False)
cpdef list tt_random_gaussian(list target_ranks, tuple shape=(2,)):
    cdef list compl_target_ranks = [1] + target_ranks + [1]
    return tt_normalise(
        [np.divide(1, l_n * np.prod(shape) * l_np1) * np.random.randn(l_n, *shape, l_np1) for l_n, l_np1 in
         zip(compl_target_ranks[:-1], compl_target_ranks[1:])])


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef cnp.ndarray[cnp.int64_t, ndim=1] symmetric_powers_of_two(int length):
    if length <= 0:
        return np.array([], dtype=np.int64)

    cdef int half = length // 2
    cdef int i
    cdef cnp.ndarray[cnp.int64_t, ndim=1] result = np.empty(length, dtype=np.int64)
    for i in range(half):
        result[i] = 1LL << (i + 1)

    if length % 2 != 0:
        result[half] = 1LL << (half + 1)

    for i in range(half):
        result[length - 1 - i] = result[i]
            
    return result


@cython.boundscheck(False)
@cython.wraparound(False)
def add_kick_rank(cnp.ndarray[double, ndim=2] u,
                   cnp.ndarray[double, ndim=2] v,
                   int r_add=2):
    cdef int old_r = u.shape[1]
    cdef int M = u.shape[0]
    cdef int N = v.shape[1]

    # Add random Gaussian kick
    cdef cnp.ndarray[double, ndim=2] uk = np.random.randn(M, r_add)

    # Concatenate and QR
    cdef cnp.ndarray[double, ndim=2] concat = np.ascontiguousarray(np.concatenate((u, uk), axis=1))
    cdef tuple qr_result = scp.linalg.qr(concat, mode='economic', check_finite=False)
    cdef cnp.ndarray[double, ndim=2] u_new = qr_result[0]
    cdef cnp.ndarray[double, ndim=2] Rmat = qr_result[1]

    # Adjust v
    cdef cnp.ndarray[double, ndim=2] v_new = Rmat[:, :old_r] @ v
    cdef int new_rank = u_new.shape[1]

    return u_new, v_new, new_rank
