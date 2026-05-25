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
import scipy as scp
from opt_einsum import contract as einsum

cnp.import_array() # Initialize NumPy C-API

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
    cdef int i, idx = np.random.randint(0, n)
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
cdef tuple swap_cores(cnp.ndarray core_a, cnp.ndarray core_b, double eps):
    cdef cnp.ndarray tensor_contraction, transposed_contraction, reshaped_matrix
    cdef cnp.ndarray u, s, v, core_a_new, core_b_new
    cdef int r_pruned

    if core_a.ndim == 3:
        tensor_contraction = np.tensordot(core_a, core_b, axes=([2], [0]))
        transposed_contraction = tensor_contraction.transpose((0, 2, 1, 3))
        reshaped_matrix = transposed_contraction.reshape(
            core_a.shape[0] * core_b.shape[1], -1)

        u, s, v = scp.linalg.svd(reshaped_matrix, full_matrices=False, check_finite=False, overwrite_a=True, lapack_driver="gesvd")
        r_pruned = prune_singular_vals(s, eps)

        core_a_new = np.reshape(u[:, :r_pruned] * s[:r_pruned].reshape(1, -1),
                                (core_a.shape[0], core_b.shape[1], -1))
        core_b_new = np.reshape(v[:r_pruned, :],
                                (-1, core_a.shape[1], core_b.shape[2]))

        return core_a_new, core_b_new
    tensor_contraction = np.tensordot(core_a, core_b, axes=([3], [0]))
    transposed_contraction = tensor_contraction.transpose((0, 3, 4, 1, 2, 5))
    reshaped_matrix = transposed_contraction.reshape(
        core_a.shape[0] * core_b.shape[1] * core_b.shape[2], -1)

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
cpdef double tt_inner_prod(list train_1_tt, list train_2_tt):
    cdef cnp.ndarray[double, ndim=2] result
    cdef cnp.ndarray temp_result
    cdef cnp.ndarray core1, core2
    cdef tuple core_pair
    result = np.array([[1.0]], dtype=np.double)
    for core_pair in zip(train_1_tt, train_2_tt):
        core1, core2 = core_pair
        temp_result = np.tensordot(result, core1, axes=([0], [0]))
        if core1.ndim == 4:
            result = np.tensordot(temp_result, core2, axes=([0, 1, 2], [0, 1, 2]))
        else:
            result = np.tensordot(temp_result, core2, axes=([0, 1], [0, 1]))

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