import autograd.numpy as np
from autograd import grad
import scipy as scp
from src.ops import *
from tqdm import tqdm
import copy

PHI = np.array([[1, 1],
                [1, -1]], dtype=float).reshape(1, 2, 2, 1)

PHI_INV = np.array([[1 / 2, 1 / 2],
                    [1 / 2, -1 / 2]], dtype=float).reshape(1, 2, 2, 1)

I = np.array([[1, 0],
              [0, 1]]).reshape(1, 2, 2, 1)


def tt_identity(dim):
    return [I for _ in range(dim)]


def tt_zeros(dim, shape=(2,)):
    return [np.zeros((1, *shape, 1)) for _ in range(dim)]


def phi(num_bonds):
    """ Bonds "num_bonds" PHI cores together """
    tensor = PHI
    new_PHI = PHI
    for _ in range(2 ** (num_bonds) - 1):
        new_PHI = np.expand_dims(new_PHI, (0, 1))
        tensor = np.kron(np.expand_dims(tensor, (-1, -2)), new_PHI)
    return tensor


def phi_inv(num_bonds):
    """ Bonds "num_bonds" PHI cores together """
    tensor = PHI_INV
    new_PHI = PHI_INV
    for _ in range(2 ** (num_bonds) - 1):
        new_PHI = np.expand_dims(new_PHI, (0, 1))
        tensor = np.kron(np.expand_dims(tensor, (-1, -2)), new_PHI)
    return tensor


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


def tt_one(dim, shape=(2,)):
    """ Returns an all-one tensor of dimension 2**dim """
    return [np.ones((1, *shape, 1)) for _ in range(dim)]


def tt_rl_orthogonalise(train_tt: List[np.array]):
    for idx in range(len(train_tt) - 1, 0, -1):
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
    shape_ip1 = train_tt[i+1].shape
    [q_T, r] = scp.linalg.qr(train_tt[i].reshape(-1, shape_i[-1]), overwrite_a=True, mode='economic', check_finite=False)
    train_tt[i] = q_T.reshape(shape_i[0], *shape_i[1:-1], -1)
    r_ip1 = train_tt[i].shape[-1]
    train_tt[i+1] = (r @ train_tt[i + 1].reshape(r.shape[-1], -1)).reshape(r_ip1, *shape_ip1[1:-1], -1)
    return train_tt


def core_backward_orthogonalise(i, train_tt):
    shape_i = train_tt[i].shape
    shape_im1 = train_tt[i - 1].shape
    [q_T, r] = scp.linalg.qr(train_tt[i].reshape(train_tt[i].shape[0], -1).T, overwrite_a=True, mode='economic', check_finite=False)
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
    train_1_tt = tt_rl_contraction(train_1_tt, train_2_tt)
    return tt_swap_all(train_1_tt)


def tt_lr_random_orthogonalise(train_tt: List[np.array], target_ranks: List[int]) -> List[np.array]:
    if len(train_tt) > 1:
        tt_gaussian = tt_random_gaussian(target_ranks, shape=train_tt[0].shape[1:-1])
        return _tt_lr_random_orthogonalise(train_tt, tt_gaussian)
    return train_tt


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


def tt_rank_reduce(train_tt: List[np.array], err_bound=None):
    """ Might reduce TT-rank """
    train_tt = tt_rl_orthogonalise(train_tt)
    if err_bound is None:
        err_bound = np.divide(1, 2 ** (len(train_tt) + 2))
    rank = 1
    for idx, tt_core in enumerate(train_tt[:-1]):
        idx_shape = tt_core.shape
        next_idx_shape = train_tt[idx + 1].shape
        k = len(idx_shape) - 1
        U, S, V_T = np.linalg.svd(train_tt[idx].reshape(rank * np.prod(idx_shape[1:k], dtype=int), -1))
        non_sing_eig_idxs = np.asarray(S >= min(np.max(S), err_bound)).nonzero()
        S = S[non_sing_eig_idxs]
        next_rank = len(S)
        U = U[:, non_sing_eig_idxs]
        V_T = V_T[non_sing_eig_idxs, :]
        train_tt[idx] = U.reshape(rank, *idx_shape[1:-1], next_rank)
        train_tt[idx + 1] = (
            np.diag(S) @ V_T @ train_tt[idx + 1].reshape(V_T.shape[-1], -1)
        ).reshape(next_rank, *next_idx_shape[1:-1], -1)
        rank = next_rank
    return train_tt


def tt_ranks(train_tt):
    return [t.shape[-1] for t in train_tt[:-1]]


def tt_rank_retraction(train_tt: List[np.array], upper_ranks: List[int]):
    """ Might reduce TT-rank """
    train_tt = tt_rl_orthogonalise(train_tt)
    rank = 1
    for idx, upper_rank in enumerate(upper_ranks):
        idx_shape = train_tt[idx].shape
        next_idx_shape = train_tt[idx + 1].shape
        k = len(idx_shape) - 1
        U, S, V_T = np.linalg.svd(train_tt[idx].reshape(rank * np.prod(idx_shape[1:k], dtype=int), -1))
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


def tt_svd(tensor: np.array) -> List[np.array]:
    """ Converts a tensor into a tensor train """
    shape = tensor.shape
    rank = 1
    cores = []
    for i in range(len(shape) - 1):
        A = tensor.reshape(rank * shape[i], -1)
        U, S, V_T = np.linalg.svd(A)
        non_sing_eig_idxs = np.asarray(np.abs(S) > 1e-5).nonzero()
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
    return sum([
        np.kron(core_1[(slice(None),) + i], core_2[(slice(None),) + i])
        for i in product(*([list(range(s)) for s in core_1.shape[1:-1]]))
    ])


def tt_inner_prod(train_1_tt: List[np.array], train_2_tt: List[np.array]) -> float:
    """
    Computes the inner product between two tensor trains
    """
    return np.sum(
        safe_multi_dot([_tt_core_collapse(core_1, core_2) for core_1, core_2 in zip(train_1_tt, train_2_tt)])
    )


def tt_to_tensor(tt_train):
    tensor = tt_train[0]
    for core in tt_train[1:]:
        tensor = np.tensordot(tensor, core, axes=(-1, 0))
    return np.sum(tensor, axis=(0, -1))


def tt_xnor(train_1_tt: List[np.array], train_2_tt: List[np.array]) -> List[np.array]:
    """
    Produces the truth table result tensor
    """
    train_xnor_tt = tt_hadamard(train_1_tt, train_2_tt)
    return tt_rank_reduce(train_xnor_tt)


def tt_xor(train_1_tt: List[np.array], train_2_tt: List[np.array]) -> List[np.array]:
    return tt_xnor(tt_neg(train_1_tt), train_2_tt)


def tt_and(train_1_tt: List[np.array], train_2_tt: List[np.array]) -> List[np.array]:
    train_1_tt = tt_scale(0.5, train_1_tt)
    mul_tt = tt_hadamard(train_1_tt, train_2_tt)
    train_2_tt = tt_scale(0.5, train_2_tt)
    half = tt_scale(-0.5, tt_one(len(train_1_tt)))
    train_and_tt = tt_add(tt_add(half, mul_tt), tt_add(train_1_tt, train_2_tt))
    return tt_rank_reduce(train_and_tt)


def tt_or(train_1_tt: List[np.array], train_2_tt: List[np.array]) -> List[np.array]:
    return tt_neg(tt_and(tt_neg(train_1_tt), tt_neg(train_2_tt)))


def tt_neg(train_tt: List[np.array]) -> List[np.array]:
    return tt_scale(-1, train_tt)


def tt_scale(alpha, train_tt):
    idx = np.random.randint(low=0, high=len(train_tt))
    return sum([train_tt[:idx], [alpha * train_tt[idx]], train_tt[idx + 1:]], [])


def tt_normalise(train_tt, radius=1):
    return tt_scale(np.divide(radius, np.sqrt(tt_inner_prod(train_tt, train_tt))), train_tt)


def tt_add_column(matrix_train_tt, column_tt, i):
    binary_i = bin(i)[2:]
    binary_i = '0' * (len(matrix_train_tt) - len(binary_i)) + binary_i
    index = [(np.expand_dims(np.zeros_like(c), 1), np.expand_dims(c, 1)) if bool(int(b)) else (
        np.expand_dims(c, 1), np.expand_dims(np.zeros_like(c), 1)) for
             b, c in zip(binary_i, column_tt)]
    column_tt = [np.concatenate(tup, axis=1) for tup in index]
    return tt_add(matrix_train_tt, column_tt)


def tt_conjugate_gradient(linear_op: List[np.array], tt_train: List[np.array], num_iter=10, tol=1e-7):
    x = [np.zeros((1, 2, 1)) for _ in range(len(tt_train))]
    r = p = tt_train
    r_2 = tt_inner_prod(r, r)
    for _ in range(num_iter):
        prev_r_2 = r_2
        l_op_p = tt_matrix_vec_mul(linear_op, p)
        p_rest = tt_inner_prod(p, l_op_p)
        alpha = np.divide(prev_r_2, p_rest)
        r = tt_rl_orthogonalise(tt_add(r, tt_scale(-alpha, l_op_p, idx=np.random.randint(low=0, high=len(p)))))
        r_2 = tt_inner_prod(r, r)
        beta = np.divide(r_2, prev_r_2)
        x = tt_rl_orthogonalise(tt_add(x, tt_scale(alpha, p, idx=np.random.randint(low=0, high=len(p)))))
        if np.less_equal(r_2, tol):
            break
        p = tt_rl_orthogonalise(tt_add(r, tt_scale(beta, p, idx=np.random.randint(low=0, high=len(p)))))
    return tt_rank_reduce(x)


def tt_swap_all(tt_train: List[np.array]):
    return [np.swapaxes(c, 0, -1) for c in reversed(tt_train)]


def tt_max_eigentensor(matrix_tt: List[np.array], num_iter=10, tol=1e-3):
    normalisation = np.sqrt(tt_inner_prod(matrix_tt, matrix_tt))
    matrix_tt = tt_scale(np.divide(1, normalisation), matrix_tt)
    eig_vec_tt = tt_normalise([np.random.randn(1, 2, 1) for _ in range(len(matrix_tt))])
    matrix_tt = tt_rank_reduce(matrix_tt)
    norm_2 = np.inf
    for i in range(num_iter):
        prev_norm_2 = norm_2
        eig_vec_tt = tt_matrix_vec_mul(matrix_tt, eig_vec_tt)
        norm_2 = tt_inner_prod(eig_vec_tt, eig_vec_tt)
        eig_vec_tt = tt_scale(np.divide(1, np.sqrt(norm_2)), eig_vec_tt)
        if np.less_equal(np.abs(norm_2 - prev_norm_2), tol):
            break
    prev_eig_vec = eig_vec_tt
    eig_vec_tt = tt_matrix_vec_mul(matrix_tt, eig_vec_tt)
    eig_val = tt_inner_prod(prev_eig_vec, eig_vec_tt)
    return tt_normalise(eig_vec_tt), normalisation * eig_val


def _tt_mat_core_collapse(core_op: np.array, core: np.array) -> np.array:
    indices_op = (slice(None),) * (len(core_op.shape) - 2)
    indices_core = (slice(None),) + (None,) * (len(core_op.shape) - len(core.shape))
    return sum([
        np.kron(core_op[indices_op + (i,)], core[indices_core + (i,)])
        for i in range(core.shape[1])
    ])


def _tt_matrix_vec_mul(matrix_tt: List[np.array], vec_tt: List[np.array]) -> List[np.array]:
    split_idx = np.argmax([len(c.shape) for c in matrix_tt])
    op_length = len(matrix_tt) - split_idx
    full_cores = [_tt_mat_core_collapse(core_op, core) for core_op, core in
                  zip(matrix_tt[split_idx:], vec_tt[-op_length:])]
    left_overs = len(vec_tt) - op_length
    if left_overs > 0:
        half_core = safe_multi_dot(
            [_tt_core_collapse(core_op, core) for core_op, core in
             zip(matrix_tt[split_idx - left_overs:split_idx], vec_tt[:-op_length])]
        )
        full_cores[0] = np.einsum("ab, bce -> ace", half_core, full_cores[0])
    full_cores = matrix_tt[:split_idx - left_overs] + full_cores
    return full_cores


def tt_matrix_vec_mul(matrix_tt, vec_tt: List[np.array], bound=None) -> List[np.array]:
    vec_tt = _tt_matrix_vec_mul(matrix_tt, vec_tt)
    return tt_rank_reduce(vec_tt, bound)


def tt_randomised_matrix_vec_mul(matrix_tt, vec_tt: List[np.array], ranks) -> List[np.array]:
    return tt_lr_random_orthogonalise(_tt_matrix_vec_mul(matrix_tt, vec_tt), target_ranks=ranks)


def _tt_mat_mat_collapse(mat_core_1, mat_core_2):
    return sum([
        np.kron(mat_core_1[:, None, i], mat_core_2[:, :, None, i])
        for i in range(mat_core_2.shape[2])
    ])


def tt_mat_mat_mul(matrix_tt_1, matrix_tt_2):
    split_idx = np.argmax([len(c.shape) for c in matrix_tt_1])
    half_core = safe_multi_dot([
        _tt_core_collapse(core_op_1, core_op_2) for core_op_1, core_op_2 in
        zip(matrix_tt_1[:split_idx], matrix_tt_2[:split_idx])
    ])
    full_cores = [
        _tt_mat_mat_collapse(core_op_1, core_op_2) for core_op_1, core_op_2 in
        zip(matrix_tt_1[split_idx:], matrix_tt_2[split_idx:])
    ]
    if len(half_core) > 0:
        full_cores[0] = np.einsum("ab, bcde -> acde", half_core, full_cores[0])
    return full_cores


def tt_transpose(matrix_tt):
    """
    Constructs the transpose of the linear op
    """
    split_idx = np.argmax([len(c.shape) for c in matrix_tt])
    return matrix_tt[:split_idx] + [np.swapaxes(c, axis1=1, axis2=2) for c in matrix_tt[split_idx:]]


def tt_gram(matrix_tt):
    """ Constructs the gram tensor for a linear op"""
    matrix_tt_t = tt_transpose(matrix_tt)
    gram = tt_mat_mat_mul(matrix_tt_t, matrix_tt)
    gram = tt_rank_reduce(gram)
    return gram


def tt_trace(matrix_tt):
    I = tt_identity(len(matrix_tt))
    return tt_inner_prod(matrix_tt, I)


def tt_min_eigentensor(matrix_tt: List[np.array], num_iter=10, tol=1e-3):
    """
    Only for symmetric matrices
    """
    n = len(matrix_tt)
    normalisation = np.sqrt(tt_inner_prod(matrix_tt, matrix_tt))
    matrix_tt = tt_scale(-np.divide(1, normalisation), matrix_tt)
    identity = tt_identity(n)
    identity = tt_scale(2, identity)
    matrix_tt = tt_add(identity, matrix_tt)
    matrix_tt = tt_rank_reduce(matrix_tt, err_bound=0)
    eig_vec_tt = tt_normalise([np.random.randn(1, 2, 1) for _ in range(n)])
    norm_2 = np.inf
    for i in range(num_iter):
        prev_norm_2 = norm_2
        eig_vec_tt = tt_matrix_vec_mul(matrix_tt, eig_vec_tt, bound=0)
        norm_2 = tt_inner_prod(eig_vec_tt, eig_vec_tt)
        eig_vec_tt = tt_scale(np.divide(1, np.sqrt(norm_2)), eig_vec_tt)
        if np.less_equal(np.abs(norm_2 - prev_norm_2), tol):
            break
    prev_eig_vec = eig_vec_tt
    eig_vec_tt = tt_matrix_vec_mul(matrix_tt, eig_vec_tt, bound=0)
    eig_val = tt_inner_prod(prev_eig_vec, eig_vec_tt)
    return tt_normalise(eig_vec_tt), normalisation * (2 - eig_val)


def tt_randomised_min_eigentensor(matrix_tt: List[np.array], num_iter=10, tol=1e-3):
    ranks = [2 ** i for i in range(int(np.ceil((len(matrix_tt) + 1) / 2)))]
    target_ranks = ranks + list(reversed(ranks[:len(matrix_tt) + 1 - len(ranks)]))
    gaussian_tt = [
        np.divide(1, l_n * 2 * l_np1) * np.random.randn(l_n, 2, l_np1)
        for i, (l_n, l_np1) in enumerate(zip(target_ranks[:-1], target_ranks[1:]))
    ]
    return _tt_randomised_min_eigentensor(matrix_tt, gaussian_tt, num_iter, tol)


def _tt_randomised_min_eigentensor(matrix_tt: List[np.array], gaussian_tt, num_iter=10, tol=1e-3):
    """
    Only for symmetric matrices
    """
    n = len(matrix_tt)
    normalisation = np.sqrt(tt_inner_prod(matrix_tt, matrix_tt))
    matrix_tt = tt_scale(-np.divide(1, normalisation), matrix_tt)
    identity = tt_identity(n)
    identity = tt_scale(2, identity)
    matrix_tt = tt_rank_reduce(tt_add(identity, matrix_tt), err_bound=tol)
    eig_vec_tt = tt_normalise([np.random.randn(1, 2, 1) for _ in range(n)])
    norm_2 = np.inf
    for i in range(num_iter):
        prev_norm_2 = norm_2
        eig_vec_tt = _tt_lr_random_orthogonalise(_tt_matrix_vec_mul(matrix_tt, eig_vec_tt), gaussian_tt)
        norm_2 = tt_inner_prod(eig_vec_tt, eig_vec_tt)
        eig_vec_tt = tt_scale(np.divide(1, np.sqrt(norm_2)), eig_vec_tt)
        if np.less_equal(np.abs(norm_2 - prev_norm_2), tol):
            break
    prev_eig_vec = eig_vec_tt
    eig_vec_tt = _tt_lr_random_orthogonalise(_tt_matrix_vec_mul(matrix_tt, eig_vec_tt), gaussian_tt)
    eig_val = tt_inner_prod(prev_eig_vec, eig_vec_tt)
    return tt_normalise(eig_vec_tt), normalisation * (2 - eig_val)


def tt_outer_product(train_1_tt, train_2_tt):
    outer = [np.kron(np.expand_dims(c_1, 1), np.expand_dims(c_2, 2)) for c_1, c_2 in zip(train_1_tt, train_2_tt)]
    return outer


def _tt_phi_core(core: np.array):
    return sum([
        np.kron(
            np.expand_dims(core[(slice(None),) + i], list(range(1, 1 + len(i)))),
            phi(len(i) - 1)[(slice(None),) + sum(zip(i, [slice(None)] * len(i)), ())]
        ) for i in product(*([[0, 1]] * (len(core.shape) - 2)))])


def tt_walsh_op(train_tt: List[np.array]) -> List[np.array]:
    """
    Produces the -1, 1 binary tensor from a Fourier representation
    """
    return [_tt_phi_core(core) for core in train_tt]


def _tt_phi_core_inv(core: np.array):
    return sum([
        np.kron(
            np.expand_dims(core[(slice(None),) + i], list(range(1, 1 + len(i)))),
            phi_inv(len(i) - 1)[(slice(None),) + sum(zip(i, [slice(None)] * len(i)), ())]
        ) for i in product(*([[0, 1]] * (len(core.shape) - 2)))])


def tt_walsh_op_inv(train_tt: List[np.array]) -> List[np.array]:
    """
    Produces the Fourier representation of a binary -1, 1 tensor
    """
    return [_tt_phi_core_inv(core) for core in train_tt]


def tt_binary_round(train_tt, num_iter=30, tol=1e-10):
    # FIXME: Easier to just use ALS
    for _ in range(num_iter):
        tt_train_p2 = tt_rank_reduce(tt_hadamard(train_tt, train_tt))
        boolean_poly = tt_walsh_op_inv(tt_train_p2)
        if np.less_equal(np.abs(tt_inner_prod(boolean_poly, boolean_poly) - 1), tol):
            break
        tt_train_p3 = tt_rank_reduce(tt_hadamard(tt_train_p2, train_tt))
        tt_update = tt_scale(-0.5, tt_train_p3)
        next_tt_train = tt_scale(1.5, train_tt)
        train_tt = tt_rank_reduce(tt_add(next_tt_train, tt_update))
    return train_tt


def tt_random_graph(target_ranks):
    matrix = tt_random_gaussian([int(np.ceil(r / 2)) for r in target_ranks], shape=(2, 2))
    matrix_t = tt_transpose(matrix)
    symmetric_matrix = tt_rank_reduce(tt_add(matrix, matrix_t))
    I_tensor = sum([break_core_bond(I) for _ in symmetric_matrix], [])
    mask = tt_add(tt_one(len(I_tensor)), tt_scale(-1, I_tensor))
    symmetric_matrix = sum([break_core_bond(core) for core in symmetric_matrix], [])
    symmetric_matrix = tt_walsh_op(symmetric_matrix)
    symmetric_matrix = tt_normalise(symmetric_matrix)
    symmetric_matrix = tt_binary_round(symmetric_matrix)
    symmetric_matrix = tt_hadamard(symmetric_matrix, mask)
    symmetric_matrix = tt_add(symmetric_matrix, tt_scale(-1, I_tensor))
    return tt_rank_reduce([core_bond(c_1, c_2) for c_1, c_2 in zip(symmetric_matrix[:-1:2], symmetric_matrix[1::2])],
                          err_bound=0)


def tt_matrix_to_matrix(matrix_tt):
    tensor = tt_to_tensor(matrix_tt)
    n = len(tensor.shape)
    axes = [i for i in range(0, n - 1, 2)] + [i for i in range(1, n, 2)]
    return np.transpose(tensor, axes).reshape(np.prod(tensor.shape[:n // 2]), -1)


def tt_vec_to_vec(vec_tt):
    tensor = tt_to_tensor(vec_tt)
    return tensor.reshape(-1, 1)


def _tt_op_core_collapse(op_core: np.array, core: np.array) -> np.array:
    return sum([
        np.kron(op_core[:, :, i, j], core[:, None, i, j])
        for (i, j) in product([0, 1], [0, 1])
    ])


def tt_linear_op(linear_op_tt, matrix_tt):
    split_idx = np.argmax([len(c.shape) for c in linear_op_tt])
    factor = safe_multi_dot(
        [_tt_core_collapse(core_1, core_2) for core_1, core_2 in zip(linear_op_tt[:split_idx], matrix_tt[:split_idx])])
    full_cores = [_tt_op_core_collapse(core_1, core_2) for core_1, core_2 in
                  zip(linear_op_tt[split_idx:], matrix_tt[split_idx:])]
    if len(factor) > 0:
        full_cores[0] = np.einsum("ab, bcd -> acd", factor, full_cores[0])
    return full_cores


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


def _core_mask(core, i, j):
    mask = np.zeros_like(core)
    mask[:, i, j] += core[:, i, j]
    return np.expand_dims(mask, 1)


def tt_mask_to_linear_op(train_tt):
    """
    Converts a matrix that serves as a mask via a Hadamard product to a linear operator
    """
    return tt_rank_reduce(
        [np.concatenate([_core_mask(c, i, j) for (i, j) in product([0, 1], [0, 1])], axis=1) for c in train_tt]
    )


def tt_argmax(train_tt, p=1):
    """
    FIXME:Acknowledgement missing here
    """
    orth_tt_train = tt_rl_orthogonalise(copy.copy(train_tt))
    k = 2 * (len(orth_tt_train))
    d = len(orth_tt_train)
    p0 = p / d

    G = orth_tt_train[-1]
    r1, n, r2 = G.shape

    I = np.arange(n).reshape(-1, 1)
    Q = G.reshape(r1, n)

    Q *= 2 ** p0

    for G in orth_tt_train[:-1][::-1]:
        r1, n, r2 = G.shape
        Q = np.einsum('qir,rk->qik', G, Q, optimize='optimal')
        Q = Q.reshape(r1, -1)
        I_l = np.kron(np.arange(n).reshape(-1, 1), np.ones((I.shape[0], 1)))
        I_r = np.kron(np.ones((n, 1)), I)
        I = np.hstack((I_l, I_r))
        q_max = np.max(np.abs(Q))
        norms = np.sum((Q / q_max) ** 2, axis=0)
        ind = np.argsort(norms)[:-(k + 1):-1]
        I = I[ind, :]
        Q = Q[:, ind]
        Q *= 2 ** p0

    return I[0]


def commutation_matrix(m, n):
    return np.eye(m * n)[np.arange(m * n).reshape((m, n), order="F").T.ravel(order="F"), :]


def vec(Y):
    return Y.reshape(-1, 1, order='F')


def mat(y, shape):
    return y.reshape(*shape, order='F')

import time

def _tt_burer_monteiro_grad(A_22, A_33, A_44, C_00, C_01, C_10, _, V_00, V_01, V_10, V_11):
    m, n = C_00.shape
    orig_p, orig_q = V_00.shape
    p = orig_p ** 2
    q = orig_q ** 2
    max_s = max(q*m, n*p)
    I = np.eye(max_s)
    indices = np.arange(max_s)
    K_np = I[indices[:n*p].reshape((n, p), order="F").ravel(), :n * p]
    K_qm = I[indices[:q*m].reshape((q, m), order="F").ravel(), :q * m]
    K_orig_qp = I[indices[:orig_q * orig_p].reshape((orig_q, orig_p), order="F").ravel(), :orig_q * orig_p]
    S = (
        np.einsum('ijt, jl->ilt', A_22.T.reshape(p, m * q, n, order="F"), K_qm).reshape(p * q, m * n, order="F")
        + np.einsum('ijt, jl->ilt', A_33.T.reshape(m, n*p, q, order="F"), K_np).reshape(m*n, p*q, order="F").T
    )
    pair_1 = np.kron(V_00, V_00) + np.kron(V_10, V_10)
    pair_2 = np.kron(V_01, V_00) + np.kron(V_11, V_10)
    pair_3 = np.kron(V_00, V_01) + np.kron(V_10, V_11)

    D_1 = (
        np.trace((pair_1 @ A_44.reshape(q, q*p**2)).reshape(p, p, p, q, order="F"), axis1=0, axis2=2)
        + np.trace((A_44.reshape(p*q**2, p) @ pair_1).reshape(p, q, q, q, order="F"), axis1=1, axis2=3)
    ).reshape(p*q, 1, order="F") + S @ C_00.reshape(m*n, 1, order="F")
    D_2 = (
        np.trace((pair_2 @ A_44.reshape(q, q*p**2)).reshape(p, p, p, q, order="F"), axis1=0, axis2=2)
        + np.trace((A_44.reshape(p*q**2, p) @ pair_2).reshape(p, q, q, q, order="F"), axis1=1, axis2=3)
    ).reshape(p*q, 1, order="F") + S @ C_01.reshape(m*n, 1, order="F")
    D_3 = (
        np.trace((pair_3 @ A_44.reshape(q, q*p**2)).reshape(p, p, p, q, order="F"), axis1=0, axis2=2)
        + np.trace((A_44.reshape(p*q**2, p) @ pair_3).reshape(p, q, q, q, order="F"), axis1=1, axis2=3)
    ).reshape(p*q, 1, order="F") + S @ C_10.reshape(m*n, 1, order="F")

    H_0 = D_1.reshape(orig_p, q*orig_p, order="F") @ np.einsum('ijt, jl->ilt', K_orig_qp.reshape(orig_q*orig_p, orig_p, orig_q), V_00).reshape(orig_p * q, orig_q, order="F")
    H_1 = D_2.reshape(orig_p, q*orig_p, order="F") @ np.einsum('ijt, jl->ilt', K_orig_qp.reshape(orig_q*orig_p, orig_p, orig_q), V_01).reshape(orig_p * q, orig_q, order="F")
    G_0 = np.einsum('ijt, lj->ilt', K_orig_qp.reshape(orig_q*orig_p, orig_q, orig_p, order="F"), V_00).reshape(orig_q * p, orig_p).T @ D_1.reshape(orig_q * p, orig_q, order="F")
    G_1 = np.einsum('ijt, lj->ilt', K_orig_qp.reshape(orig_q*orig_p, orig_q, orig_p, order="F"), V_01).reshape(orig_q * p, orig_p).T @ D_3.reshape(orig_q * p, orig_q, order="F")

    return G_0 + H_0 + H_1 + G_1


def _tt_bm_core_wise(matrix_tt, factor_tt, A_22, A_33, A_44, idx, is_block=False, lr=0.5, num_swps=20, gamma=0.9, tol=1e-5):
    xr_i, _, _, xr_ip1 = factor_tt[idx].shape
    local_lr = lr

    if 0 < idx < len(factor_tt) - 1:
        C_00 = matrix_tt[idx][:, 0, 0]
        C_01 = matrix_tt[idx][:, 0, 1]
        C_10 = matrix_tt[idx][:, 1, 0]
        C_11 = matrix_tt[idx][:, 1, 1]
        V_00 = factor_tt[idx][:, 0, 0]
        V_01 = factor_tt[idx][:, 0, 1]
        V_10 = factor_tt[idx][:, 1, 0]
        V_11 = factor_tt[idx][:, 1, 1]
    else:
        xr_i = max(xr_i, xr_ip1)
        xr_ip1 = max(xr_i, xr_ip1)
        C_00 = np.diag(matrix_tt[idx][:, 0, 0].flatten())
        C_10 = np.diag(matrix_tt[idx][:, 1, 0].flatten())
        C_01 = np.diag(matrix_tt[idx][:, 0, 1].flatten())
        C_11 = np.diag(matrix_tt[idx][:, 1, 1].flatten())
        V_00 = np.diag(factor_tt[idx][:, 0, 0].flatten())
        V_10 = np.diag(factor_tt[idx][:, 1, 0].flatten())
        V_01 = np.diag(factor_tt[idx][:, 0, 1].flatten())
        V_11 = np.diag(factor_tt[idx][:, 1, 1].flatten())

    v_00_grad = 0
    v_01_grad = 0
    v_10_grad = 0
    v_11_grad = 0

    vec_01 = np.zeros(xr_ip1 ** 2)
    vec_10 = np.zeros(xr_ip1 ** 2)

    for swp in range(num_swps):
        V_00_nest = V_00 - gamma * v_00_grad
        V_01_nest = V_01 - gamma * v_01_grad
        V_10_nest = V_10 - gamma * v_10_grad
        V_11_nest = V_11 - gamma * v_11_grad
        vec_00 = _tt_burer_monteiro_grad(A_22, A_33, A_44, C_00, C_01, C_10, C_11, V_00_nest, V_01_nest, V_10_nest, V_11_nest)
        vec_11 = _tt_burer_monteiro_grad(A_22, A_33, A_44, C_11, C_10, C_01, C_00, V_11_nest, V_10_nest, V_01_nest, V_00_nest)

        if idx != 0 or not is_block:
            vec_01 = _tt_burer_monteiro_grad(A_22, A_33, A_44, C_11, C_10, C_01, C_00, V_01_nest, V_00_nest, V_11_nest, V_10_nest)
            vec_10 = _tt_burer_monteiro_grad(A_22, A_33, A_44, C_00, C_01, C_10, C_11, V_10_nest, V_11_nest, V_00_nest, V_01_nest)

        v_00_grad = gamma*v_00_grad + local_lr * vec_00.reshape(xr_i, xr_ip1)
        v_01_grad = gamma*v_01_grad + local_lr * vec_01.reshape(xr_i, xr_ip1)
        v_10_grad = gamma*v_10_grad + local_lr * vec_10.reshape(xr_i, xr_ip1)
        v_11_grad = gamma*v_11_grad + local_lr * vec_11.reshape(xr_i, xr_ip1)
        V_00 -= v_00_grad
        V_01 -= v_01_grad
        V_10 -= v_10_grad
        V_11 -= v_11_grad

        local_err = (np.linalg.norm(v_00_grad)+np.linalg.norm(v_01_grad)+np.linalg.norm(v_10_grad)+np.linalg.norm(v_11_grad))/4
        local_lr *= 0.99
        if np.less_equal(local_err, tol):
            break

    if 0 < idx < len(factor_tt) - 1:
        factor_tt[idx][:, 0, 0, :] = V_00
        factor_tt[idx][:, 0, 1, :] = V_01
        factor_tt[idx][:, 1, 0, :] = V_10
        factor_tt[idx][:, 1, 1, :] = V_11
    else:
        ax = 0 if idx == 0 else -1
        factor_tt[idx][:, 0, 0, :] = np.expand_dims(np.diagonal(V_00.reshape(xr_i, xr_ip1)), axis=ax)
        factor_tt[idx][:, 0, 1, :] = np.expand_dims(np.diagonal(V_01.reshape(xr_i, xr_ip1)), axis=ax)
        factor_tt[idx][:, 1, 0, :] = np.expand_dims(np.diagonal(V_10.reshape(xr_i, xr_ip1)), axis=ax)
        factor_tt[idx][:, 1, 1, :] = np.expand_dims(np.diagonal(V_11.reshape(xr_i, xr_ip1)), axis=ax)

    return factor_tt, lr


def _adjust_diff(matrix_tt, factor_tt, idx):
    comp_tt_idx = _tt_mat_mat_collapse(factor_tt[idx], np.swapaxes(factor_tt[idx], axis1=1, axis2=2))
    if 0 < idx < len(factor_tt) - 1:
        diff_core_idx = _block_diag_tensor(matrix_tt[idx], comp_tt_idx)
    elif idx == 0:
        diff_core_idx = np.concatenate((matrix_tt[idx], comp_tt_idx), axis=-1)
    else:
        diff_core_idx = np.concatenate((matrix_tt[idx], comp_tt_idx), axis=0)
    return _tt_core_collapse(diff_core_idx, diff_core_idx)


def tt_burer_monteiro_factorisation(psd_tt, solution_tt=None, is_block=False, num_swps=20, max_iter=20, tol=1e-5):
    train_tt = tt_scale(-1, psd_tt)
    target_ranks = [int(np.ceil(np.sqrt(r)))+1 for r in tt_ranks(train_tt)]
    if solution_tt is None:
        solution_tt = tt_random_gaussian(target_ranks, shape=(2, 2))
    else:
        add_on_ranks = [max(r - c_r, 0) for c_r, r in zip(tt_ranks(solution_tt), target_ranks)]
        solution_tt = tt_add(solution_tt, tt_random_gaussian(add_on_ranks, shape=(2, 2)))
    solution_tt = tt_rl_orthogonalise(solution_tt)
    if is_block:
        solution_tt[0][:, 0, 1] = 0
        solution_tt[0][:, 1, 0] = 0
    lr = 0.2
    prev_err = 100
    comp_tt = tt_mat_mat_mul(solution_tt, tt_transpose(solution_tt))
    diff = [_tt_core_collapse(c, c) for c in tt_add(train_tt, comp_tt)]
    train_shapes = [(train_tt[0].shape[-1], train_tt[0].shape[-1])] + [(c.shape[0], c.shape[-1]) for c in train_tt[1:-1]] + [(train_tt[-1].shape[0], train_tt[-1].shape[0])]
    solution_shapes = [(solution_tt[0].shape[-1]**2, solution_tt[0].shape[-1]**2)] + [(c.shape[0]**2, c.shape[-1]**2) for c in solution_tt[1:-1]] + [(solution_tt[-1].shape[0]**2, solution_tt[-1].shape[0]**2)]
    index_set = [
        (
            (
                np.array([[i + (n + q) * j for i in range(q)] for j in range(n)]).flatten() + n,
                np.array([[i + (m + p) * j for i in range(p)] for j in range(m)]).flatten() + m
            ),
            (
                np.array([[i + (n + q) * j for i in range(n)] for j in range(q)]).flatten() + n * (n + q),
                np.array([[i + (m + p) * j for i in range(m)] for j in range(p)]).flatten() + m * (m + p)
            ),
            (
                np.array([[i + (n + q) * j for i in range(q)] for j in range(q)]).flatten() + n * (n + q + 1),
                np.array([[i + (m + p) * j for i in range(p)] for j in range(p)]).flatten() + m * (m + p + 1)
            )
        )
        for (m, n), (p, q) in zip(train_shapes, solution_shapes)
    ]
    terminal_idx = len(solution_tt) - 1
    for iteration in range(max_iter):
        left_contraction = 1
        for k in range(len(solution_tt) - 1):
            solution_tt = core_forward_orthogonalise(k, solution_tt)
            diff[k] = _adjust_diff(train_tt, solution_tt, k)
            left_contraction = np.dot(left_contraction, diff[k])
            if k+1 != terminal_idx:
                right_contraction = safe_multi_dot(diff[k + 2:])
                A_22 = right_contraction.reshape(-1, 1)[index_set[k+1][0][0]] @ left_contraction.reshape(1, -1)[:, index_set[k+1][0][1]]
                A_33 = right_contraction.reshape(-1, 1)[index_set[k+1][1][0]] @ left_contraction.reshape(1, -1)[:, index_set[k+1][1][1]]
                A_44 = right_contraction.reshape(-1, 1)[index_set[k+1][2][0]] @ left_contraction.reshape(1, -1)[:, index_set[k+1][2][1]]
            else:
                A_22 = np.diag(left_contraction.flatten()[index_set[k + 1][0][1]])
                A_33 = np.diag(left_contraction.flatten()[index_set[k + 1][1][1]])
                A_44 = np.diag(left_contraction.flatten()[index_set[k + 1][2][1]])
            solution_tt, lr = _tt_bm_core_wise(train_tt, solution_tt, A_22, A_33, A_44, k + 1, is_block=is_block, lr=lr, num_swps=num_swps, tol=0.1 * tol)
        right_contraction = 1
        for k in range(terminal_idx, 0, -1):
            solution_tt = core_backward_orthogonalise(k, solution_tt)
            diff[k] = _adjust_diff(train_tt, solution_tt, k)
            right_contraction = np.dot(diff[k], right_contraction)
            if k-1 != 0:
                left_contraction = safe_multi_dot(diff[:k - 1]).reshape(1, -1)
                A_22 = right_contraction.reshape(-1, 1)[index_set[k - 1][0][0]] @ left_contraction.reshape(1, -1)[:, index_set[k - 1][0][1]]
                A_33 = right_contraction.reshape(-1, 1)[index_set[k - 1][1][0]] @ left_contraction.reshape(1, -1)[:, index_set[k - 1][1][1]]
                A_44 = right_contraction.reshape(-1, 1)[index_set[k - 1][2][0]] @ left_contraction.reshape(1, -1)[:, index_set[k - 1][2][1]]
            else:
                A_22 = np.diag(right_contraction.flatten()[index_set[k - 1][0][1]])
                A_33 = np.diag(right_contraction.flatten()[index_set[k - 1][1][1]])
                A_44 = np.diag(right_contraction.flatten()[index_set[k - 1][2][1]])
            solution_tt, lr = _tt_bm_core_wise(train_tt, solution_tt, A_22, A_33, A_44, k - 1, is_block=is_block, lr=lr, num_swps=num_swps, tol=0.1 * tol)

        diff[0] = _adjust_diff(train_tt, solution_tt, 0)
        err = (diff[0] @ right_contraction).item()
        lr = min(0.99 * (prev_err / err) * lr, 0.2)
        prev_err = err
        print(f"Error: {err}, {lr}")
        if np.less_equal(err, tol):
            print(f"Converged in {iteration} iterations")
            break

    return solution_tt, prev_err


def tt_to_ring(train_tt):
    first_core_top_row = np.hstack((np.diag(train_tt[0][:, 0, 0].flatten()), np.diag(train_tt[0][:, 0, 1].flatten())))
    first_core_bottom_row = np.hstack(
        (np.diag(train_tt[0][:, 1, 0].flatten()), np.diag(train_tt[0][:, 1, 1].flatten())))
    first_core = np.vstack((first_core_top_row, first_core_bottom_row))
    train_tt[0] = first_core

    last_core_top_row = np.hstack((np.diag(train_tt[-1][:, 0, 0].flatten()), np.diag(train_tt[-1][:, 0, 1].flatten())))
    last_core_bottom_row = np.hstack(
        (np.diag(train_tt[-1][:, 1, 0].flatten()), np.diag(train_tt[-1][:, 1, 1].flatten())))
    last_core = np.vstack((last_core_top_row, last_core_bottom_row))
    train_tt[-1] = last_core

    return train_tt


def _core_op_from_matrix(core):
    op_core = np.zeros((core.shape[0], 4, *core.shape[1:]))
    op_core[:, 0, 0, 0] = core[:, 0, 0]
    op_core[:, 0, 0, 1] = core[:, 1, 0]
    op_core[:, 1, 0, 0] = core[:, 0, 1]
    op_core[:, 1, 0, 1] = core[:, 1, 1]
    op_core[:, 2, 1, 0] = core[:, 0, 0]
    op_core[:, 2, 1, 1] = core[:, 1, 0]
    op_core[:, 3, 1, 0] = core[:, 0, 1]
    op_core[:, 3, 1, 1] = core[:, 1, 1]
    return op_core


def tt_op_from_tt_matrix(matrix_tt):
    """
    Converts a matrix_tt for matrix multiplication into a linear_op_tt
    """
    return [_core_op_from_matrix(c) for c in matrix_tt]


def tt_adjoint(linear_op_tt):
    return [np.swapaxes(c, axis1=2, axis2=3) for c in linear_op_tt]


def tt_vec(matrix_tt):
    return [c.reshape(c.shape[0], np.prod(c.shape[1:-1]), c.shape[-1]) for c in matrix_tt]


def tt_mat(matrix_tt, shape=(2, 2)):
    return [c.reshape(c.shape[0], 2, 2, c.shape[-1]) for c in matrix_tt]


def tt_op_to_mat(op_tt):
    return [c.reshape(c.shape[0], 4, 4, c.shape[-1]) for c in op_tt]
