import numpy as np
import scipy as scp

from src.ops import *
from cy_src.tt_ops_cy import *
from opt_einsum import contract as einsum


def E(i, j):
    E = np.zeros((1, 2, 2, 1))
    E[:, i, j] += 1
    return E


def tt_random_binary(target_ranks: List[int], shape=(2,)):
    """
    Only has non-negative entries
    """
    target_ranks = [1] + target_ranks + [1]
    return tt_normalise([np.random.randint(low=0, high=2, size=(l_n, *shape, l_np1)) for l_n, l_np1 in
                         zip(target_ranks[:-1], target_ranks[1:])])

def tt_random_gaussian(target_ranks: List[int], shape=(2,)):
    target_ranks = [1] + target_ranks + [1]
    return tt_normalise(
        [np.divide(1, l_n * np.prod(shape) * l_np1) * np.random.randn(l_n, *shape, l_np1) for l_n, l_np1 in
         zip(target_ranks[:-1], target_ranks[1:])])

def tt_rl_orthogonalise(train_tt: List[np.array]):
    dim = len(train_tt)
    if dim == 1:
        return train_tt
    for idx in range(dim - 1, 0, -1):
        shape_p1 = train_tt[idx].shape
        shape = train_tt[idx - 1].shape
        Q_T, R = np.linalg.qr(train_tt[idx].reshape(shape_p1[0], -1).T)
        train_tt[idx] = Q_T.T.reshape(-1, *shape_p1[1:-1], shape_p1[-1])
        train_tt[idx - 1] = (train_tt[idx - 1].reshape(-1, R.shape[-1]) @ R.T).reshape(-1, *shape[1:-1],
                                                                                       train_tt[idx].shape[0])
    return train_tt

def tt_orthogonalise_idx(train_tt: List[np.array], idx):
    if idx == 0:
        return tt_rl_orthogonalise(train_tt)
    if idx == len(train_tt) - 1:
        return tt_lr_orthogonalise(train_tt)
    tt_train_fidx = tt_lr_orthogonalise(train_tt[:idx + 1])
    tt_train_uidx = tt_rl_orthogonalise([tt_train_fidx[-1]] + train_tt[idx + 1:])
    return tt_train_fidx[:-1] + tt_train_uidx

def tt_lr_orthogonalise(train_tt: List[np.array]):
    train_tt = tt_swap_all(train_tt)
    train_tt = tt_rl_orthogonalise(train_tt)
    train_tt = tt_swap_all(train_tt)
    return train_tt


def core_forward_orthogonalise(i, train_tt):
    shape_i = train_tt[i].shape
    shape_ip1 = train_tt[i + 1].shape
    [q_T, r] = np.linalg.qr(train_tt[i].reshape(-1, shape_i[-1]))
    train_tt[i] = q_T.reshape(shape_i[0], *shape_i[1:-1], -1)
    r_ip1 = train_tt[i].shape[-1]
    train_tt[i + 1] = (r @ train_tt[i + 1].reshape(r.shape[-1], -1)).reshape(r_ip1, *shape_ip1[1:-1], -1)
    return train_tt


def core_backward_orthogonalise(i, train_tt):
    shape_i = train_tt[i].shape
    shape_im1 = train_tt[i - 1].shape
    [q_T, r] = np.linalg.qr(train_tt[i].reshape(train_tt[i].shape[0], -1).T)
    train_tt[i] = q_T.T.reshape(-1, *shape_i[1:-1], shape_i[-1])
    r_im1 = train_tt[i].shape[0]
    train_tt[i - 1] = (
        train_tt[i - 1].reshape(-1, r.shape[-1]) @ r.T
    ).reshape(-1, *shape_im1[1:-1], r_im1)
    return train_tt


def tt_bond_at(train_tt, idx):
    if idx != -1:
        train_tt = train_tt[:idx] + [core_bond(train_tt[idx], train_tt[idx + 1])] + train_tt[idx + 2:]
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
        Q_T, _ = np.linalg.qr(core_y)  # R_i * 2 x unknown
        train_tt[i] = Q_T.reshape(*train_tt[i].shape[:-1], -1)  # R_i * 2 x unknown
        core_m = Q_T.T @ core_z  # unknown x R_{i+1}
        train_tt[i + 1] = (
            core_m @ train_tt[i + 1].reshape(shape_i1[0], -1)
        ).reshape(-1, *shape_i1[1:])  # unknown x 2 * R_{i+2}
    return train_tt


def tt_rank_reduce(train_tt: List[np.array], eps=1e-18):
    """ Might reduce TT-rank """
    dim = len(train_tt)
    eps = eps / np.sqrt(dim-1)
    if dim == 1 or np.all(np.array(tt_ranks(train_tt))==1):
        return train_tt
    train_tt = tt_rl_orthogonalise(train_tt)
    rank = 1
    for idx, tt_core in enumerate(train_tt[:-1]):
        idx_shape = tt_core.shape
        next_idx_shape = train_tt[idx + 1].shape
        k = len(idx_shape) - 1
        u, s, v_t = scp.linalg.svd(train_tt[idx].reshape(rank * np.prod(idx_shape[1:k], dtype=int), -1), full_matrices=False, check_finite=False)
        next_rank = prune_singular_vals(s, eps)
        s = s[:next_rank]
        u = u[:, :next_rank]
        v_t = v_t[:next_rank, :]
        train_tt[idx] = u.reshape(rank, *idx_shape[1:-1], next_rank)
        train_tt[idx + 1] = (
            np.diag(s) @ v_t @ train_tt[idx + 1].reshape(v_t.shape[-1], -1)
        ).reshape(next_rank, *next_idx_shape[1:-1], -1)
        rank = next_rank
    return train_tt


def tt_rank_retraction(train_tt: List[np.array], upper_ranks: List[int]):
    """ Might reduce TT-rank """
    train_tt = tt_rl_orthogonalise(train_tt)
    rank = 1
    for idx, upper_rank in enumerate(upper_ranks):
        idx_shape = train_tt[idx].shape
        next_idx_shape = train_tt[idx + 1].shape
        U, S, V_T = scip.linalg.svd(train_tt[idx].reshape(rank * np.prod(idx_shape[1:-1], dtype=int), -1), full_matrices=False)
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
        U, S, V_T = scip.linalg.svd(A, full_matrices=False, check_finite=False)
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
            [core[tuple([slice(None)] + [i] * (len(core.shape) - 2))] for i, core in zip(indices, train_tt)]))


def _block_diag_tensor(tensor_1: np.array, tensor_2: np.array) -> np.array:
    """
    For internal use: Concatenates two tensors to a block diagonal tensor
    """
    column_1 = np.concatenate((tensor_1, np.zeros((tensor_2.shape[0],) + tensor_1.shape[1:])), axis=0)
    column_2 = np.concatenate((np.zeros((tensor_1.shape[0],) + tensor_2.shape[1:]), tensor_2), axis=0)
    return np.concatenate((column_1, column_2), axis=-1)


def tt_add(train_1_tt: List[np.array], train_2_tt: List[np.array]) -> List[np.array]:
    """
    Adds two tensor trains
    """
    if len(train_1_tt) > 1:
        return [np.concatenate((train_1_tt[0], train_2_tt[0]), axis=-1)] + \
            [_block_diag_tensor(core_1, core_2) for core_1, core_2 in zip(train_1_tt[1:-1], train_2_tt[1:-1])] + \
            [np.concatenate((train_1_tt[-1], train_2_tt[-1]), axis=0)]
    return [train_1_tt[0] + train_2_tt[0]]


def tt_sub(train_1_tt: List[np.array], train_2_tt: List[np.array]) -> List[np.array]:
    return tt_add(train_1_tt, tt_scale(-1, train_2_tt))


def _tt_train_kron(core_1: np.array, core_2: np.array) -> np.array:
    """
    For internal use: Computes the kronecker product between two TT-cores with appropriate dimensional
    expansion
    """
    core_shape_length = len(core_1.shape)
    axes = list(range(1, core_shape_length - 1))
    layers = [
        np.kron(
            np.expand_dims(core_1[(slice(None),) + i], axis=axes),
            np.expand_dims(core_2[(slice(None),) + i], axis=axes)
        ) for i in product(*([[0, 1]] * (core_shape_length - 2)))
    ]
    return np.concatenate(layers, axis=1).reshape(
        (core_1.shape[0] * core_2.shape[0],) + core_1.shape[1:-1] + (core_1.shape[-1] * core_2.shape[-1],)
    )


def tt_hadamard(train_1_tt: List[np.array], train_2_tt: List[np.array]) -> List[np.array]:
    """
    Computes the hadamard product/pointwise multiplication of two tensor trains
    """
    return [_tt_train_kron(core_1, core_2) for core_1, core_2 in zip(train_1_tt, train_2_tt)]


def _tt_core_collapse(core_1: np.array, core_2: np.array) -> np.array:
    # TODO: Switch kron for einsum
    return np.sum([
        np.kron(core_1[(slice(None),) + i], core_2[(slice(None),) + i])
        for i in product(*([list(range(s)) for s in core_1.shape[1:-1]]))
    ], axis=0)


def tt_inner_prod(train_1_tt: List[np.array], train_2_tt: List[np.array]) -> float:
    return np.sum(
        np.linalg.multi_dot([_tt_core_collapse(core_1, core_2) for core_1, core_2 in zip(train_1_tt, train_2_tt)])
    )


def tt_to_tensor(tt_train):
    tensor = tt_train[0]
    for core in tt_train[1:]:
        tensor = np.tensordot(tensor, core, axes=(-1, 0))
    return np.sum(tensor, axis=(0, -1))


def tt_normalise(train_tt, radius=1):
    return tt_scale(np.divide(radius, np.sqrt(tt_inner_prod(train_tt, train_tt))), train_tt)


def _tt_mat_core_collapse(core_op: np.array, core: np.array) -> np.array:
    indices_op = (slice(None),) * (len(core_op.shape) - 2)
    indices_core = (slice(None),) + (None,) * (len(core_op.shape) - len(core.shape))
    return sum([
        np.kron(core_op[indices_op + (i,)], core[indices_core + (i,)])
        for i in range(core.shape[1])
    ])


def tt_matrix_vec_mul(matrix_tt: List[np.array], vec_tt: List[np.array]) -> List[np.array]:
    return [_tt_mat_core_collapse(core_op, core) for core_op, core in zip(matrix_tt, vec_tt)]


def prune_singular_vals(s, eps):
    if np.linalg.norm(s) == 0.0:
        return 1

    if eps <= 0.0:
        return s.size

    sc = np.cumsum(np.abs(s[::-1]) ** 2)[::-1]
    R = np.argmax(sc < eps ** 2)
    R = max(R, 1)
    R = s.size if sc[-1] > eps ** 2 else R

    return R


def swap_cores(core_a, core_b, eps):
    if len(core_a.shape) == 3 and len(core_b.shape) == 3:
        supercore = einsum("rms,snR->rnmR", core_a, core_b)
        u, s, v = scip.linalg.svd(np.reshape(supercore, (core_a.shape[0] * core_b.shape[1], -1)), full_matrices=False, check_finite=False)
        u = u @ np.diag(s)
        r = prune_singular_vals(s, eps)
        u = u[:, :r]
        v = v[:r, :]
        return np.reshape(u, (core_a.shape[0], core_b.shape[1], -1)), np.reshape(v,(-1, core_a.shape[1], core_b.shape[2]))
    elif len(core_a.shape) == 4 and len(core_b.shape) == 4:
        supercore = einsum("rmas,snbR->rnbmaR", core_a, core_b)
        u, s, v = scip.linalg.svd(np.reshape(supercore, (core_a.shape[0] * core_b.shape[1] * core_b.shape[2], -1)), full_matrices=False, check_finite=False)
        u = u @ np.diag(s)
        r = prune_singular_vals(s, eps)
        u = u[:, :r]
        v = v[:r, :]
        return np.reshape(u, (core_a.shape[0], core_b.shape[1], core_b.shape[2], -1)), np.reshape(v, (-1, core_a.shape[1], core_a.shape[2], core_b.shape[3]))
    else:
        raise Exception("The cores must be wither 3d or 4d tensors.")


def tt_fast_matrix_vec_mul(matrix_tt: List[np.array], vec_tt: List[np.array], eps=1e-18) -> List[np.array]:
    """ https://arxiv.org/pdf/2410.19747 """
    dim = len(matrix_tt)
    eps = eps / np.sqrt(dim-1)

    cores = [np.transpose(c, (2, 1, 0)) for c in vec_tt[::-1]]
    for i in range(dim):
        cores[0] = einsum("mabk,kbn->man", matrix_tt[dim - i - 1], cores[0])

        if i != dim - 1:
            for j in range(i, -1, -1):
                cores[j], cores[j + 1] = swap_cores(cores[j], cores[j + 1], eps)

    return cores

def _tt_mat_mat_collapse(mat_core_1, mat_core_2):
    return sum([
        np.kron(mat_core_1[:, :, None, i], mat_core_2[:, None, i])
        for i in range(mat_core_2.shape[2])
    ])


def tt_mat_mat_mul(matrix_tt_1, matrix_tt_2):
    return [
        _tt_mat_mat_collapse(core_op_1, core_op_2) for core_op_1, core_op_2 in
        zip(matrix_tt_1, matrix_tt_2)
    ]

def tt_fast_mat_mat_mul(matrix_tt_1, matrix_tt_2, eps=1e-18):
    dim= len(matrix_tt_1)
    eps = eps / np.sqrt(dim-1)

    cores = [np.transpose(c, (3, 1, 2, 0)) for c in matrix_tt_2[::-1]]
    for i in range(dim):
        cores[0] = einsum("mabk,kbcn->macn", matrix_tt_1[dim - i - 1], cores[0])

        if i != dim - 1:
            for j in range(i, -1, -1):
                cores[j], cores[j + 1] = swap_cores(cores[j], cores[j + 1], eps)

    return cores


def tt_fast_hadammard(train_tt_1, train_tt_2, eps=1e-18):
    dim = len(train_tt_1)
    eps = eps / np.sqrt(dim-1)

    if len(train_tt_1[0].shape) == 4 and len(train_tt_2[0].shape) == 4:

        cores = [np.transpose(c, (3, 1, 2, 0)) for c in train_tt_2[::-1]]
        for i in range(dim):
            cores[0] = einsum("maAk,kbBn,AB,ab->maAn", train_tt_1[dim - i - 1], cores[0],
                                   np.eye(train_tt_1[dim - i - 1].shape[1], dtype=cores[0].dtype),
                                   np.eye(train_tt_1[dim - i - 1].shape[1], dtype=cores[0].dtype))

            if i != dim - 1:
                for j in range(i, -1, -1):
                    cores[j], cores[j + 1] = swap_cores(cores[j], cores[j + 1], eps)

        return cores
    else:

        cores = [np.transpose(c, (2, 1, 0)) for c in train_tt_2[::-1]]
        for i in range(dim):
            cores[0] = einsum("mak,kbn,ab->man", train_tt_1[dim - i - 1], cores[0],
                                   np.eye(train_tt_1[dim - i - 1].shape[1], dtype=cores[0].dtype))

            if i != dim - 1:
                for j in range(i, -1, -1):
                    cores[j], cores[j + 1] = swap_cores(cores[j], cores[j + 1], eps)

        return cores


def tt_kron(matrix_tt_1, matrix_tt_2):
    result_tt = [np.kron(c_1[:, None, None], c_2[:, :, :, None, None]) for c_1, c_2 in zip(matrix_tt_1, matrix_tt_2)]
    result_tt = sum([break_core_bond(c) for c in result_tt], [])
    return result_tt


def tt_gram(matrix_tt):
    """ Constructs the gram tensor for a linear op"""
    matrix_tt_t = tt_transpose(matrix_tt)
    gram = tt_mat_mat_mul(matrix_tt_t, matrix_tt)
    gram = tt_rank_reduce(gram)
    return gram


def tt_trace(matrix_tt):
    I = tt_identity(len(matrix_tt))
    return tt_inner_prod(matrix_tt, I)


def tt_random_graph(dim, max_rank, max_iter=100):
    if dim == 1:
        d = np.round(np.random.rand())
        return [np.array([[np.round(np.random.rand()), d], [d, np.round(np.random.rand())]]).reshape(1, 2, 2, 1)]
    max_rank = max_rank
    perm_cores = [
        np.array([[1.0, 0.0], [0.0, 0.0]]).reshape(1, 2, 2, 1),
        np.array([[0.0, 1.0], [0.0, 0.0]]).reshape(1, 2, 2, 1),
        np.array([[0.0, 0.0], [1.0, 0.0]]).reshape(1, 2, 2, 1),
        np.array([[0.0, 0.0], [0.0, 1.0]]).reshape(1, 2, 2, 1)
    ]
    unique_rows = set()
    graph = tt_zero_matrix(dim)
    for _ in range(max_iter):
        row = np.random.randint(0, 3, dim)
        if (1 in row) or (2 in row):
            if tuple(row) not in unique_rows:
                cores = [perm_cores[i] for i in row]
                cores = tt_add(cores, tt_transpose(cores))
                graph = tt_add(graph, cores)
                graph = tt_rank_reduce(graph)
            unique_rows.add(tuple(row))
            row_t = np.where(row == 1, 2, np.where(row == 2, 1, row))
            unique_rows.add(tuple(row_t))
        if np.max(tt_ranks(graph)) >= max_rank:
            break

    return graph


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
    return tt_mat(tt_svd(tensor, err_bound))


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


def tt_vec(matrix_tt):
    return sum([break_core_bond(c) for c in matrix_tt], [])


def tt_mat(vec_tt):
    return [einsum("abc, cde -> abde", c_1, c_2) for c_1, c_2 in zip(vec_tt[:-1:2], vec_tt[1::2])]


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

def tt_diag(vec_tt):
    identity = np.eye(2)
    basis = [einsum("ij, rjR -> rijR", identity, c) for c in vec_tt]
    return tt_rank_reduce(basis)

def tt_diagonal(matrix_tt):
    return [np.transpose(np.diagonal(c,  axis1=1, axis2=2), (0, 2, 1)) for c in matrix_tt]

def tt_sum(*args):
    sum_tt = args[0]
    for arg in args[1:]:
        sum_tt =  tt_rank_reduce(tt_add(sum_tt, arg))
    return sum_tt
