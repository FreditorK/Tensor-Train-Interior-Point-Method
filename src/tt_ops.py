import scipy as scp
from typing import *

from cy_src.tt_ops_cy import *
from cy_src.lgmres_cy import lgmres, MatVecWrapper, IneqMatVecWrapper
from functools import reduce
from functools import lru_cache
from opt_einsum import contract_expression
from opt_einsum import contract as einsum

def E(i, j):
    E = np.zeros((1, 2, 2, 1))
    E[:, i, j] += 1
    return E


@lru_cache(maxsize=2048)
def get_contract_expr_cached(equation: str, shapes: tuple):
    return contract_expression(equation, *shapes, optimize='greedy')

def cached_einsum(equation: str, *operands):
    expr = get_contract_expr_cached(equation, tuple(op.shape for op in operands))
    return expr(*operands)

def tt_rl_orthogonalise_py(train_tt: List[np.array]):
    dim = len(train_tt)
    if dim == 1:
        return train_tt
    for i in range(dim - 1, -1, -1):
        shape_i = train_tt[i].shape
        shape_im1 = train_tt[i - 1].shape
        q_T, r = scp.linalg.qr(train_tt[i].reshape(train_tt[i].shape[0], -1).T, check_finite=False, mode="economic")
        train_tt[i] = q_T.T.reshape(-1, *shape_i[1:-1], shape_i[-1])
        train_tt[i - 1] = (
                train_tt[i - 1].reshape(-1, r.shape[-1]) @ r.T
        ).reshape(-1, *shape_im1[1:-1], train_tt[i].shape[0])
    return train_tt

def tt_lr_orthogonalise_py(train_tt: List[np.array]):
    train_tt = tt_swap_all(train_tt)
    train_tt = tt_rl_orthogonalise(train_tt)
    train_tt = tt_swap_all(train_tt)
    return train_tt


def tt_rl_contraction(train_1_tt: List[np.array], train_2_tt: List[np.array]):
    new_cores = [
        train_1_tt[-1].reshape(train_1_tt[-1].shape[0], -1) @ train_2_tt[-1].reshape(train_2_tt[-1].shape[0], -1).T]
    for core_1, core_2 in zip(train_1_tt[-2:0:-1], train_2_tt[-2:0:-1]):
        core_w = new_cores[-1]
        core_z = core_1.reshape(-1, core_w.shape[0]) @ core_w
        new_cores.append(core_z.reshape(core_1.shape[0], -1) @ core_2.reshape(core_2.shape[0], -1).T)
    return new_cores[::-1]


def tt_lr_contraction(train_1_tt: List[np.array], train_2_tt: List[np.array]):
    train_1_tt = tt_swap_all(train_1_tt)
    train_2_tt = tt_swap_all(train_2_tt)
    train_1_tt = tt_rl_contraction(train_1_tt, train_2_tt)
    return tt_swap_all(train_1_tt)


def tt_lr_random_orthogonalise(train_tt: List[np.array], target_ranks: List[int]) -> List[np.array]:
    if len(train_tt) > 1:
        tt_gaussian = tt_random_gaussian(target_ranks, shape=train_tt[0].shape[1:-1])
        return _tt_lr_random_orthogonalise(train_tt, tt_gaussian)
    return train_tt


def tt_rl_random_orthogonalise(train_tt: List[np.array], target_ranks: List[int]) -> List[np.array]:
    if len(train_tt) > 1:
        tt_gaussian = tt_swap_all(tt_random_gaussian(target_ranks, shape=train_tt[0].shape[1:-1]))
        train_tt = tt_swap_all(train_tt)
        return tt_swap_all(_tt_lr_random_orthogonalise(train_tt, tt_gaussian))
    return train_tt


def _tt_rl_random_orthogonalise(train_tt, gaussian_tt):
    train_tt = tt_swap_all(train_tt)
    gaussian_tt = tt_swap_all(gaussian_tt)
    return tt_swap_all(_tt_lr_random_orthogonalise(train_tt, gaussian_tt))


def _tt_lr_random_orthogonalise(train_tt, gaussian_tt):
    gaussian_contractions = tt_rl_contraction(train_tt, gaussian_tt)
    for i, core_w in enumerate(gaussian_contractions):
        shape_i1 = train_tt[i + 1].shape
        core_z = train_tt[i].reshape(-1, shape_i1[0])  # R_i * 2 x R_{i+1}
        core_y = core_z @ core_w  # R_i * 2 x target_r
        Q_T, _ = scp.linalg.qr(core_y, check_finite=False, mode="economic", overwrite_a=True)  # R_i * 2 x unknown
        train_tt[i] = Q_T.reshape(*train_tt[i].shape[:-1], -1)  # R_i * 2 x unknown
        core_m = Q_T.T @ core_z  # unknown x R_{i+1}
        train_tt[i + 1] = (
            core_m @ train_tt[i + 1].reshape(shape_i1[0], -1)
        ).reshape(-1, *shape_i1[1:])  # unknown x 2 * R_{i+2}
    return train_tt


def tt_rank_reduce_py(train_tt: List[np.array], eps=1e-18):
    """ Might reduce TT-rank """
    dim = len(train_tt)
    ranks = np.array([1] + tt_ranks(train_tt) + [1])
    if dim == 1 or np.all(ranks==1):
        return train_tt
    eps = (eps / np.sqrt(dim - 1))
    train_tt = tt_rl_orthogonalise(train_tt)
    rank = 1
    for idx, tt_core in enumerate(train_tt[:-1]):
        idx_shape = tt_core.shape
        next_idx_shape = train_tt[idx + 1].shape
        u, s, v_t = scp.linalg.svd(
            train_tt[idx].reshape(rank * np.prod(idx_shape[1:len(idx_shape) - 1], dtype=int), -1),
            full_matrices=False,
            check_finite=False,
            overwrite_a=True,
            lapack_driver="gesvd"
        )
        next_rank = prune_singular_vals(s, eps)
        train_tt[idx] = u[:, :next_rank].reshape(rank, *idx_shape[1:-1], next_rank)
        train_tt[idx + 1] = (
            s[:next_rank].reshape(-1, 1) * v_t[:next_rank, :] @ train_tt[idx + 1].reshape(v_t.shape[-1], -1)
        ).reshape(next_rank, *next_idx_shape[1:-1], -1)
        rank = next_rank
    return train_tt


def tt_rank_retraction(train_tt: List[np.array], upper_ranks: List[int]):
    """ Might reduce TT-rank """
    train_tt = tt_rl_orthogonalise_py(train_tt)
    rank = 1
    for idx, upper_rank in enumerate(upper_ranks):
        idx_shape = train_tt[idx].shape
        next_idx_shape = train_tt[idx + 1].shape
        U, S, V_T = scp.linalg.svd(train_tt[idx].reshape(rank * np.prod(idx_shape[1:-1], dtype=int), -1), full_matrices=False, check_finite=False, overwrite_a=True, lapack_driver="gesvd")
        abs_S = np.abs(S)
        next_rank = min(upper_rank, len(abs_S > 0))
        non_sing_eig_idxs = np.argpartition(abs_S, -next_rank)[-next_rank:]
        S = S[non_sing_eig_idxs]
        U = U[:, non_sing_eig_idxs]
        V_T = V_T[non_sing_eig_idxs, :]
        train_tt[idx] = U.reshape(rank, *idx_shape[1:-1], next_rank)
        train_tt[idx + 1] = (np.diag(S) @ V_T @ train_tt[idx + 1].reshape(V_T.shape[-1], -1)).reshape(next_rank,
                                                                                                      *next_idx_shape[
                                                                                                       1:-1],
                                                                                                      -1)
        rank = next_rank
    return train_tt


def tt_svd(tensor: np.array, err_bound=1e-18) -> List[np.array]:
    """ Converts a tensor into a tensor train """
    shape = tensor.shape
    err_bound = err_bound * np.sqrt(np.divide(tensor.reshape(1, -1) @ tensor.reshape(-1, 1), len(shape) - 1))
    rank = 1
    cores = []
    for i in range(len(shape) - 1):
        A = tensor.reshape(rank * shape[i], -1)
        U, S, V_T = scp.linalg.svd(A, full_matrices=False, check_finite=False, lapack_driver="gesvd")
        S = S.flatten()
        _, non_sing_eig_idxs = np.asarray(S >= min(np.max(S), err_bound)).nonzero()
        S = S[non_sing_eig_idxs]
        next_rank = len(S)
        U = U[:, non_sing_eig_idxs]
        V_T = V_T[non_sing_eig_idxs, :]
        G_i = U.reshape(rank, shape[i], next_rank)
        cores.append(G_i)
        tensor = np.diag(S) @ V_T
        rank = next_rank
    G_n = tensor.reshape(rank, 2, 1)
    cores.append(G_n)
    return cores


def tt_entry(train_tt: List[np.array], indices: List[int]) -> np.array:
    """
    Returns the entry of a TT-train according to the indices
    """
    return np.sum(
        np.linalg.multi_dot(
            [core[tuple([slice(None)] + [i] * (len(core.shape) - 2))] for i, core in zip(indices, train_tt)])
    )


def tt_sub(train_1_tt: List[np.array], train_2_tt: List[np.array]) -> List[np.array]:
    return tt_add(train_1_tt, tt_scale(-1, train_2_tt))

def tt_to_tensor(tt_train):
    tensor = tt_train[0]
    for core in tt_train[1:]:
        tensor = np.tensordot(tensor, core, axes=(-1, 0))
    return np.sum(tensor, axis=(0, -1))


def tt_kron(matrix_tt_1, matrix_tt_2):
    # TODO:  DO SVD right away to not be overbearing to memory
    path = [(0, 1)]
    result_tt = [einsum("rmnR, lijL -> rlminjRL", c_1, c_2, optimize=path).reshape(c_1.shape[0]*c_2.shape[0], c_1.shape[1]*c_2.shape[1], c_1.shape[2]*c_2.shape[2], c_1.shape[-1]*c_2.shape[-1]) for c_1, c_2 in zip(matrix_tt_1, matrix_tt_2)]
    return result_tt


def tt_trace(matrix_tt):
    I = tt_identity(len(matrix_tt))
    return tt_inner_prod(matrix_tt, I)


def tt_matrix_to_matrix(matrix_tt):
    if len(matrix_tt) == 1:
        return np.squeeze(matrix_tt)
    tensor = tt_to_tensor(matrix_tt)
    n = len(tensor.shape)
    axes = [i for i in range(0, n - 1, 2)] + [i for i in range(1, n, 2)]
    return np.transpose(tensor, axes).reshape(np.prod(tensor.shape[:n // 2]), -1)

def tt_matrix_svd(matrix, err_bound=1e-18):
    tensor  = np.reshape(matrix, [2]*int(np.log2(np.prod(matrix.shape))))
    n = len(tensor.shape)
    axes = sum([list(t) for t in zip([i for i in range(n // 2)], [i for i in range(n // 2, n)])], [])
    tensor = np.transpose(tensor, axes=axes)
    return tt_merge_bonds(tt_svd(tensor, err_bound))


def tt_vec_to_vec(vec_tt):
    tensor = tt_to_tensor(vec_tt)
    return tensor.reshape(-1, 1)


def tt_sketch_like(train_tt, target_ranks):
    return [
        np.divide(1, l_n * np.prod(train_tt[i].shape[1:-1]) * l_np1) * np.random.randn(l_n, *train_tt[i].shape[1:-1],
                                                                                       l_np1)
        for i, (l_n, l_np1) in enumerate(zip(target_ranks[:-1], target_ranks[1:]))
    ]


def tt_sketch(shape, target_ranks):
    return [
        np.divide(1, l_n * np.prod(shape) * l_np1) * np.random.randn(l_n, *shape, l_np1)
        for i, (l_n, l_np1) in enumerate(zip(target_ranks[:-1], target_ranks[1:]))
    ]


def _break_core_bond(core, err_bound=1e-18):
    """ Breaks up a bond between two cores """
    shape = core.shape
    k = len(shape) // 2
    A = core.reshape(np.prod(shape[:k]), -1)
    U, S, V_T = scp.linalg.svd(A, full_matrices=False, check_finite=False, overwrite_a=True)
    non_sing_eig_idxs = np.asarray(np.abs(S) > err_bound).nonzero()[0]
    if len(non_sing_eig_idxs) == 0:
        non_sing_eig_idxs = np.array([0])
    S = S[non_sing_eig_idxs]
    next_rank = len(S)
    U = U[:, non_sing_eig_idxs]
    V_T = V_T[non_sing_eig_idxs, :]
    G_i = U.reshape(*shape[:k], next_rank)
    G_ip1 = (np.diag(S) @ V_T).reshape(next_rank, *shape[k:])
    return [G_i, G_ip1]

def tt_split_bonds(matrix_tt):
    return sum([_break_core_bond(c) for c in matrix_tt], [])


def tt_merge_bonds(vec_tt):
    path = [(0, 1)]
    return [einsum("abc, cde -> abde", c_1, c_2, optimize=path) for c_1, c_2 in zip(vec_tt[:-1:2], vec_tt[1::2])]


def _tt_generalised_nystroem(tt_train, tt_gaussian_1, tt_gaussian_2):
    lr_contractions = tt_lr_contraction(tt_train, tt_gaussian_1)
    rl_contractions = tt_rl_contraction(tt_train, tt_gaussian_2)
    Ls = []
    Rs = []
    for W_L, W_R in zip(lr_contractions, rl_contractions):
        U, S, V_T = np.linalg.svd(W_L @ W_R, full_matrices=False)
        root_S_inv = np.diag(np.divide(1, np.sqrt(S)))
        L = W_R @ V_T.T @ root_S_inv
        R = root_S_inv @ U.T @ W_L
        Ls.append(L)
        Rs.append(R)
    tt_train[0] = (tt_train[0].reshape(-1, tt_train[0].shape[-1]) @ Ls[0]).reshape(*tt_train[0].shape[:-1], -1)
    for i in range(1, len(tt_train) - 1):
        tt_train[i] = (
            Rs[i - 1] @ (tt_train[i].reshape(-1, tt_train[i].shape[-1]) @ Ls[i]).reshape(tt_train[i].shape[0],
                                                                                         -1)).reshape(
            tt_train[i - 1].shape[-1], *tt_train[i].shape[1:-1], -1)
    tt_train[-1] = (Rs[-1] @ tt_train[-1].reshape(tt_train[-1].shape[0], -1)).reshape(-1, *tt_train[-1].shape[1:])
    return tt_train


def tt_generalised_nystroem(tt_train, target_ranks: List[int]) -> List[np.array]:
    if len(tt_train) > 1:
        tt_gaussian_1 = tt_random_gaussian(target_ranks, shape=tt_train[0].shape[1:-1])
        tt_gaussian_2 = tt_random_gaussian([r + 1 for r in target_ranks], shape=tt_train[0].shape[1:-1])
        return _tt_generalised_nystroem(tt_train, tt_gaussian_1, tt_gaussian_2)
    return tt_train

def tt_l2_dist(train_tt_1, train_tt_2):
    diff_tt = tt_sub(train_tt_1, train_tt_2)
    return np.sqrt(tt_inner_prod(diff_tt, diff_tt))

def tt_norm(train_tt):
    inner_prod_scal = tt_inner_prod(train_tt, train_tt)
    if inner_prod_scal > 0:
        return np.sqrt(inner_prod_scal)
    return 0.0

def tt_diag(vec_tt, eps=1e-18):
    identity = np.eye(vec_tt[0].shape[1])
    path = [(0, 1)]
    basis = [einsum("ij, rjR -> rijR", identity, c, optimize=path) for c in vec_tt]
    return tt_rank_reduce(basis, eps)

def tt_diagonal(matrix_tt):
    return [np.transpose(np.diagonal(c,  axis1=1, axis2=2), (0, 2, 1)) for c in matrix_tt]

def tt_sum(*args, op_tol=1e-18, rank_reduce=True):
    sum_tt = args[0]
    for arg in args[1:]:
        if rank_reduce:
            sum_tt =  tt_rank_reduce(tt_add(sum_tt, arg), op_tol)
        else:
            sum_tt = tt_add(sum_tt, arg)
    return sum_tt

def tt_reshape(train_tt, shape):
    if np.prod(shape) > np.prod(train_tt[0].shape[1:-1]):
        train_tt = tt_merge_cores(train_tt)
    return [c.reshape(c.shape[0], *shape, c.shape[-1]) for c in train_tt]

def tt_merge_cores(train_tt):
    path = [(0, 1)]
    if len(train_tt[0].shape[1:-1]) == 1:
        return [einsum("kir, rsK -> kisK", c_1, c_2, optimize=path) for c_1, c_2 in zip(train_tt[:-1:2], train_tt[1::2])]
    return [einsum("kijr, rsdK -> kisjdK", c_1, c_2, optimize=path) for c_1, c_2 in zip(train_tt[:-1:2], train_tt[1::2])]


def tt_entrywise_sum(train_tt):
    eq = 'ab,aijm,bijn->mn' if train_tt[0].ndim == 4 else 'ab,aim,bin->mn'
    one = np.ones((1, *train_tt[0].shape[1:-1], 1))

    result = reduce(
        lambda res, c: cached_einsum(eq, res, *(c, one)),
        train_tt,
        np.array([[1.0]])
    )

    return np.sum(result)

def tt_skew_zero_op(op_tt, eps):
    transpose_op = [np.eye(4)[[0, 2, 1, 3]].reshape(1, 4, 4, 1) for _ in op_tt]
    op_tt_t = tt_fast_mat_mat_mul(op_tt, transpose_op, eps)
    return tt_rank_reduce(tt_scale(0.5, tt_add(op_tt, op_tt_t)), eps)


def tt_IkronM(matrix_tt):
    I = np.eye(2).reshape(1, 2, 2, 1)
    path = [(0, 1)]
    return [einsum("rmnR, lijL -> rlminjRL", I, c, optimize=path).reshape(c.shape[0], 4, 4, c.shape[-1]) for c in matrix_tt]

def tt_MkronI(matrix_tt):
    I = np.eye(2).reshape(1, 2, 2, 1)
    path = [(0, 1)]
    return [einsum("rmnR, lijL -> rlminjRL", c, I, optimize=path).reshape(c.shape[0], 4, 4, c.shape[-1]) for c in matrix_tt]


def tt_diag_op(matrix_tt, eps=1e-18):
    identity = np.eye(matrix_tt[0].shape[1]*matrix_tt[0].shape[2])
    path = [(0, 1)]
    basis = [einsum("ij, rjR -> rijR", identity, c.reshape(c.shape[0], c.shape[1]*c.shape[2], c.shape[3]), optimize=path) for c in matrix_tt]
    return tt_rank_reduce(basis, eps)

def tt_tril_one_matrix(dim):
    if dim == 1:
        return [np.array([[1, 0], [1, 1]]).reshape(1, 2, 2, 1)]
    all_one = np.ones((1, 2, 2, 1))
    all_zeros = np.zeros((1, 2, 2, 1))
    return ([np.concatenate((E(1, 0), E(0, 0) + E(1, 1)), axis=-1)]
            + [np.concatenate((np.concatenate((all_one, E(1, 0)), axis=0), np.concatenate((all_zeros, E(0, 0) + E(1, 1)), axis=0)), axis=-1) for _ in range(dim-2)]
            + [np.concatenate((all_one, E(1, 0) + E(0, 0) + E(1, 1)), axis=0)]
    )

def tt_triu_one_matrix(dim):
    if dim == 1:
        return [np.array([[1, 1], [0, 1]]).reshape(1, 2, 2, 1)]
    all_one = np.ones((1, 2, 2, 1))
    all_zeros = np.zeros((1, 2, 2, 1))
    return ([np.concatenate((E(0, 1), E(0, 0) + E(1, 1)), axis=-1)]
            + [np.concatenate((np.concatenate((all_one, E(0, 1)), axis=0), np.concatenate((all_zeros, E(0, 0) + E(1, 1)), axis=0)), axis=-1) for _ in range(dim-2)]
            + [np.concatenate((all_one, E(0, 1) + E(0, 0) + E(1, 1)), axis=0)]
    )


def _null_projector(
    basis_vectors: np.ndarray,
    discarded_indices: Set[int]
) -> np.ndarray:
    dimension = len(basis_vectors)
    available_indices = list(set(range(dimension)) - discarded_indices)
    num_available = len(available_indices)

    if num_available == 0:
        return np.zeros((dimension, dimension))

    # Start with a projector onto the valid subspace.
    projector = np.eye(dimension)
    for i in discarded_indices:
        projector -= np.outer(basis_vectors[i], basis_vectors[i])

    # Determine how many random pairs to couple.
    num_couplings = np.random.randint(num_available) if num_available > 0 else 0
    if num_couplings == 0:
        return projector

    # Select source and target indices for the random couplings.
    source_indices = np.random.choice(available_indices, size=num_couplings, replace=False)
    target_indices = np.random.choice(available_indices, size=num_couplings, replace=False)

    # Add the random couplings. This operation makes the matrix no longer a strict
    # projection, but a transformation that maps a source vector to a target vector.
    for i, j in zip(source_indices, target_indices):
        projector += np.outer(basis_vectors[i], basis_vectors[j] - basis_vectors[i])

    return projector


def _diag_projector(
    basis_vectors: np.ndarray,
    discarded_indices: Set[int],
    limit=2
) -> Tuple[np.ndarray, Set[int]]:
    dimension = len(basis_vectors)
    num_couplings = np.random.randint(dimension) if dimension > 0 else 0

    source_indices = np.random.choice(dimension, size=num_couplings, replace=False)
    target_indices_1 = np.random.choice(dimension, size=num_couplings, replace=False)
    target_indices_2 = np.random.choice(dimension, size=num_couplings, replace=False)

    # Start with an identity matrix.
    projector_1 = np.eye(dimension)
    projector_2 = np.eye(dimension)
    updated_discarded_indices = discarded_indices.copy()

    # Apply transformations and update the discarded set.
    # If a source vector `i` was in the discarded set, it's now mapped to `j`,
    # so `j` effectively takes its place in the discarded set.
    for i, j_1, j_2 in zip(source_indices, target_indices_1, target_indices_2):
        if i in discarded_indices:
            if len(updated_discarded_indices) <= limit or (j_1 in discarded_indices) or (j_2 in discarded_indices):
                projector_1 += np.outer(basis_vectors[i], basis_vectors[j_1] - basis_vectors[i])
                projector_2 += np.outer(basis_vectors[i], basis_vectors[j_2] - basis_vectors[i])
                updated_discarded_indices.discard(i)
                updated_discarded_indices.add(j_1)
                updated_discarded_indices.add(j_2)
        else:
            projector_1 += np.outer(basis_vectors[i], basis_vectors[j_1] - basis_vectors[i])
            projector_2 += np.outer(basis_vectors[i], basis_vectors[j_2] - basis_vectors[i])

    return projector_1, projector_2, updated_discarded_indices

def _random_projector(basis_vectors: np.ndarray) -> np.ndarray:
    dimension = len(basis_vectors)
    if dimension == 0:
        return np.array([[]])
        
    num_couplings = np.random.randint(dimension)
    
    source_indices = np.random.choice(dimension, size=num_couplings, replace=False)
    target_indices = np.random.choice(dimension, size=num_couplings, replace=False)

    projector = np.eye(dimension)
    for i, j in zip(source_indices, target_indices):
        projector += np.outer(basis_vectors[i], basis_vectors[j] - basis_vectors[i])
        
    return projector


def _bin_rand_tril(dim: int, rank: int) -> List[np.ndarray]:
    if rank <= 0:
        return []
        
    # 1. Generate an orthonormal basis using QR decomposition of a random matrix.
    random_matrix = np.random.randn(rank, rank)
    q_matrix, _ = np.linalg.qr(random_matrix, mode='reduced')
    basis_vectors = q_matrix.T  # Each row is a basis vector

    # 2. Initialize the first core and the set of discarded indices.
    initial_indices = np.random.choice(rank, size=3, replace=True)
    # Shape: (left_bond, bond_dim, physical_dim) -> (1, 4, dimension)
    initial_core = np.zeros((1, 4, rank))
    initial_core[:, [0, 2, 3], :] = basis_vectors[initial_indices]
    
    # These indices are "used" by the fixed parts of the boundary core.
    discarded_indices = {initial_indices[0], initial_indices[2]}
    tensor_cores = [initial_core]

    if dim <= 1:
        return tensor_cores

    # 3. Generate the intermediate cores in a loop.
    for _ in range(dim - 2):
        core = np.empty((rank, 4, rank))
        core[:, 1, :] = _null_projector(basis_vectors, discarded_indices)
        core[:, 0, :], core[:, 3, :], discarded_indices = _diag_projector(basis_vectors, discarded_indices, limit=rank-1)
        core[:, 2, :] = _random_projector(basis_vectors)

        tensor_cores.append(core)

    # 4. Generate the final core, ensuring orthogonality to the discarded set.
    available_indices = list(set(range(rank)) - discarded_indices)
    num_available = len(available_indices)

    # Shape: (physical_dim, bond_dim, right_bond) -> (dimension, 4, 1)
    terminal_core = np.zeros((rank, 4, 1))
    
    # Select a random index for the one non-zero slice.
    final_fixed_index = np.random.choice(rank)

    if num_available > 0:
        # If there are available orthogonal vectors, use them for the other slices.
        ortho_indices = np.random.choice(available_indices, size=3, replace=True)
        final_indices = [ortho_indices[0], ortho_indices[1], final_fixed_index, ortho_indices[2]]
        
        # Construct the core slices from the chosen basis vectors
        # basis_vectors[final_indices] has shape (4, dimension). We transpose
        # it to fit the core's slice shape of (dimension, 4).
        terminal_core[:, :, 0] = basis_vectors[final_indices].T
    else:
        # If no orthogonal vectors are left, only set the one required slice.
        terminal_core[:, 2, 0] = basis_vectors[final_fixed_index]

    tensor_cores.append(terminal_core)

    return tensor_cores


def tt_random_graph(dim, r, eps=1e-12):
    print("====Starting Graph Sampling====")
    if r == 1:
        graph_tt = []
        for _ in range(dim-1):
            binary_numbers = np.random.randint(0, 2, size=3)
            if np.sum(binary_numbers) == 0:
                index_to_change = np.random.randint(0, 3)
                binary_numbers[index_to_change] = 1
            graph_tt.append(np.array([[binary_numbers[0], binary_numbers[1]], [binary_numbers[1], binary_numbers[2]]]).reshape(1, 2, 2, 1))

        graph_tt.append(np.array([[0, 1], [1, 0]]).reshape(1, 2, 2, 1))
        return graph_tt
    max_rank = 0
    rejection_counter = 0
    while max_rank != r:
        tril_r = 2*r
        tril = _bin_rand_tril(dim, tril_r)
        tril = tt_reshape(tril, (2, 2))
        graph_tt = tt_rank_reduce(tt_add(tril, tt_transpose(tril)), eps)
        print(f"Graph #{rejection_counter} rank: ", tt_ranks(graph_tt))
        max_rank = np.max(tt_ranks(graph_tt))
        rejection_counter += 1
    print("===Terminated Graph Sampling===")
    return graph_tt