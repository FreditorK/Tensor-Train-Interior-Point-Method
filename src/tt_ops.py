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


def tt_random_binary(target_ranks: List[int]):
    """
    Only has non-negative entries
    """
    target_ranks = [1] + target_ranks + [1]
    return tt_normalise([np.random.randint(low=0, high=2, size=(l_n, 2, l_np1)) for l_n, l_np1 in
                         zip(target_ranks[:-1], target_ranks[1:])])


def tt_random_gaussian(target_ranks: List[int], shape=(2,)):
    target_ranks = [1] + target_ranks + [1]
    return tt_normalise(
        [np.divide(1, l_n * np.prod(shape) * l_np1) * np.random.randn(l_n, *shape, l_np1) for l_n, l_np1 in
         zip(target_ranks[:-1], target_ranks[1:])])


def tt_one(dim, shape=(2,)):
    """ Returns an all-one tensor of dimension 2**dim """
    return [np.ones((1, *shape, 1)) for _ in range(dim)]


def tt_leading_one(dim):
    """ Returns a tensor of dimension 2**dim with a one entry in its 00000...00-entry"""
    return tt_atom_train(dim + 1, dim)


def tt_atom_train(idx, dim):
    """ Returns a tensor of dimension 2**dim with a one entry at the respective degree 1 monomial"""
    return [
        np.array([0, 1], dtype=float).reshape(1, 2, 1) if idx == i else np.array([1, 0], dtype=float).reshape(1, 2, 1)
        for i in range(dim)
    ]


def tt_rl_orthogonalise(tt_train: List[np.array]):
    for idx in range(len(tt_train) - 1, 0, -1):
        shape_p1 = tt_train[idx].shape
        shape = tt_train[idx - 1].shape
        Q_T, R = np.linalg.qr(tt_train[idx].reshape(shape_p1[0], -1).T)
        tt_train[idx] = Q_T.T.reshape(-1, *shape_p1[1:-1], shape_p1[-1])
        tt_train[idx - 1] = (tt_train[idx - 1].reshape(-1, R.shape[-1]) @ R.T).reshape(-1, *shape[1:-1],
                                                                                       tt_train[idx].shape[0])
    return tt_train


def tt_rl_orthogonalise_idx(tt_train: List[np.array], idx):
    if idx == 0:
        return tt_rl_orthogonalise(tt_train)
    if idx == len(tt_train) - 1:
        return tt_lr_orthogonalise(tt_train)
    tt_train_fidx = tt_lr_orthogonalise(tt_train[:idx + 1])
    tt_train_uidx = tt_rl_orthogonalise([tt_train_fidx[-1]] + tt_train[idx + 1:])
    return tt_train_fidx[:-1] + tt_train_uidx


def tt_lr_orthogonalise(tt_train: List[np.array]):
    tt_train = tt_swap_all(tt_train)
    tt_train = tt_rl_orthogonalise(tt_train)
    tt_train = tt_swap_all(tt_train)
    return tt_train


def tt_bond_at(tt_train, idx):
    if idx != -1:
        tt_train = tt_train[:idx] + [core_bond(tt_train[idx], tt_train[idx + 1])] + tt_train[idx + 2:]
    return tt_train


def tt_rl_contraction(tt_train_1: List[np.array], tt_train_2: List[np.array]):
    new_cores = [
        tt_train_1[-1].reshape(tt_train_1[-1].shape[0], -1) @ tt_train_2[-1].reshape(tt_train_2[-1].shape[0], -1).T]
    for core_1, core_2 in zip(tt_train_1[-2:0:-1], tt_train_2[-2:0:-1]):
        core_w = new_cores[-1]
        core_z = core_1.reshape(-1, core_w.shape[0]) @ core_w
        new_cores.append(core_z.reshape(core_1.shape[0], -1) @ core_2.reshape(core_2.shape[0], -1).T)
    return new_cores[::-1]


def tt_lr_contraction(tt_train_1: List[np.array], tt_train_2: List[np.array]):
    tt_train_1 = tt_swap_all(tt_train_1)
    tt_train_1 = tt_rl_contraction(tt_train_1, tt_train_2)
    return tt_swap_all(tt_train_1)


def tt_lr_random_orthogonalise(tt_train: List[np.array], target_ranks: List[int]) -> List[np.array]:
    if len(tt_train) > 1:
        tt_gaussian = tt_random_gaussian(target_ranks, shape=tt_train[0].shape[1:-1])
        return _tt_lr_random_orthogonalise(tt_train, tt_gaussian)
    return tt_train


def _tt_lr_random_orthogonalise(tt_train, tt_gaussian):
    tt_gaussian_contractions = tt_rl_contraction(tt_train, tt_gaussian)
    for i, core_w in enumerate(tt_gaussian_contractions):
        shape_i1 = tt_train[i + 1].shape
        core_z = tt_train[i].reshape(-1, shape_i1[0])  # R_i * 2 x R_{i+1}
        core_y = core_z @ core_w  # R_i * 2 x target_r
        Q_T, _ = np.linalg.qr(core_y)  # R_i * 2 x unknown
        tt_train[i] = Q_T.reshape(*tt_train[i].shape[:-1], -1)  # R_i * 2 x unknown
        core_m = Q_T.T @ core_z  # unknown x R_{i+1}
        tt_train[i + 1] = (
            core_m @ tt_train[i + 1].reshape(shape_i1[0], -1)
        ).reshape(-1, *shape_i1[1:])  # unknown x 2 * R_{i+2}
    return tt_train


def _tt_generalised_nystroem(tt_train, tt_gaussian_1, tt_gaussian_2):
    lr_contractions = tt_lr_contraction(tt_train, tt_gaussian_1)
    rl_contractions = tt_rl_contraction(tt_train, tt_gaussian_2)
    Ls = []
    Rs = []
    for W_L, W_R in zip(lr_contractions, rl_contractions):
        U, S, V_T = np.linalg.svd(W_L @ W_R)
        root_S_inv = np.diag(1 / np.sqrt(S))
        L = W_R @ V_T @ root_S_inv
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
        tt_gaussian_2 = tt_random_gaussian(target_ranks, shape=tt_train[0].shape[1:-1])
        return _tt_generalised_nystroem(tt_train, tt_gaussian_1, tt_gaussian_2)
    return tt_train


def tt_rank_reduce(tt_train: List[np.array], tt_bound=None):
    """ Might reduce TT-rank """
    tt_train = tt_rl_orthogonalise(tt_train)
    if tt_bound is None:
        tt_bound = np.divide(1, 2 ** (len(tt_train) + 2))
    rank = 1
    for idx, tt_core in enumerate(tt_train[:-1]):
        idx_shape = tt_core.shape
        next_idx_shape = tt_train[idx + 1].shape
        k = len(idx_shape) - 1
        U, S, V_T = np.linalg.svd(tt_train[idx].reshape(rank * np.prod(idx_shape[1:k], dtype=int), -1))
        non_sing_eig_idxs = np.asarray(S >= min(np.max(S), tt_bound)).nonzero()
        S = S[non_sing_eig_idxs]
        next_rank = len(S)
        U = U[:, non_sing_eig_idxs]
        V_T = V_T[non_sing_eig_idxs, :]
        tt_train[idx] = U.reshape(rank, *idx_shape[1:-1], next_rank)
        tt_train[idx + 1] = (
            np.diag(S) @ V_T @ tt_train[idx + 1].reshape(V_T.shape[-1], -1)
        ).reshape(next_rank, *next_idx_shape[1:-1], -1)
        rank = next_rank
    return tt_train


def tt_ranks(tt_train):
    return [t.shape[-1] for t in tt_train[:-1]]


def tt_rank_retraction(tt_train: List[np.array], tt_upper_ranks: List[int]):
    """ Might reduce TT-rank """
    tt_train = tt_rl_orthogonalise(tt_train)
    rank = 1
    for idx, upper_rank in enumerate(tt_upper_ranks):
        idx_shape = tt_train[idx].shape
        next_idx_shape = tt_train[idx + 1].shape
        k = len(idx_shape) - 1
        U, S, V_T = np.linalg.svd(tt_train[idx].reshape(rank * np.prod(idx_shape[1:k], dtype=int), -1))
        abs_S = np.abs(S)
        next_rank = min(upper_rank, len(abs_S > 0))
        non_sing_eig_idxs = np.argpartition(abs_S, -next_rank)[-next_rank:]
        S = S[non_sing_eig_idxs]
        U = U[:, non_sing_eig_idxs]
        V_T = V_T[non_sing_eig_idxs, :]
        tt_train[idx] = U.reshape(rank, *idx_shape[1:-1], next_rank)
        tt_train[idx + 1] = (np.diag(S) @ V_T @ tt_train[idx + 1].reshape(V_T.shape[-1], -1)).reshape(next_rank,
                                                                                                      *next_idx_shape[
                                                                                                       1:-1],
                                                                                                      -1)
        rank = next_rank
    return tt_train


def tt_svd(fourier_tensor: np.array) -> List[np.array]:
    """ Converts a tensor into a tensor train """
    shape = fourier_tensor.shape
    rank = 1
    cores = []
    for i in range(len(shape) - 1):
        A = fourier_tensor.reshape(rank * shape[i], -1)
        U, S, V_T = np.linalg.svd(A)
        non_sing_eig_idxs = np.asarray(np.abs(S) > 1e-5).nonzero()
        S = S[non_sing_eig_idxs]
        next_rank = len(S)
        U = U[:, non_sing_eig_idxs]
        V_T = V_T[non_sing_eig_idxs, :]
        G_i = U.reshape(rank, shape[i], next_rank)
        cores.append(G_i)
        fourier_tensor = np.diag(S) @ V_T
        rank = next_rank
    G_n = fourier_tensor.reshape(rank, 2, 1)
    cores.append(G_n)
    return cores


def tt_leading_entry(tt_train: List[np.array]) -> np.array:
    """
    Returns the leading entry of a TT-train
    """
    return np.sum(
        np.linalg.multi_dot([core[tuple([slice(None)] + [0] * (len(core.shape) - 2))] for core in tt_train]))


def tt_entry(tt_train: List[np.array], indices: List[int]) -> np.array:
    """
    Returns the entry of a TT-train according to the indices
    """
    return np.sum(
        np.linalg.multi_dot(
            [core[tuple([slice(None)] + [i] * (len(core.shape) - 2))] for i, core in zip(indices, tt_train)]))


def _block_diag_tensor(tensor_1: np.array, tensor_2: np.array) -> np.array:
    """
    For internal use: Concatenates two tensors to a block diagonal tensor
    """
    column_1 = np.concatenate((tensor_1, np.zeros((tensor_2.shape[0],) + tensor_1.shape[1:])), axis=0)
    column_2 = np.concatenate((np.zeros((tensor_1.shape[0],) + tensor_2.shape[1:]), tensor_2), axis=0)
    return np.concatenate((column_1, column_2), axis=-1)


def tt_add(tt_train_1: List[np.array], tt_train_2: List[np.array]) -> List[np.array]:
    """
    Adds two tensor trains
    """
    if len(tt_train_1) > 1:
        return [np.concatenate((tt_train_1[0], tt_train_2[0]), axis=-1)] + \
            [_block_diag_tensor(core_1, core_2) for core_1, core_2 in zip(tt_train_1[1:-1], tt_train_2[1:-1])] + \
            [np.concatenate((tt_train_1[-1], tt_train_2[-1]), axis=0)]
    return [tt_train_1[0] + tt_train_2[0]]


def tt_sub(tt_train_1: List[np.array], tt_train_2: List[np.array]) -> List[np.array]:
    return tt_add(tt_train_1, tt_scale(-1, tt_train_2))


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


def tt_hadamard(tt_train_1: List[np.array], tt_train_2: List[np.array]) -> List[np.array]:
    """
    Computes the hadamard product/pointwise multiplication of two tensor trains
    """
    return [_tt_train_kron(core_1, core_2) for core_1, core_2 in zip(tt_train_1, tt_train_2)]


def tt_kron(tt_train_1: List[np.array], tt_train_2: List[np.array]) -> List[np.array]:
    return tt_train_1 + tt_train_2


def _tt_core_collapse(core_1: np.array, core_2: np.array) -> np.array:
    return sum([
        np.kron(core_1[(slice(None),) + i], core_2[(slice(None),) + i])
        for i in product(*([[0, 1]] * (len(core_1.shape) - 2)))
    ])


def tt_inner_prod(tt_train_1: List[np.array], tt_train_2: List[np.array]) -> float:
    """
    Computes the inner product between two tensor trains
    """
    return np.sum(
        safe_multi_dot([_tt_core_collapse(core_1, core_2) for core_1, core_2 in zip(tt_train_1, tt_train_2)])
    )


def tt_partial_inner_prod(tt_train_1: List[np.array], tt_train_2: List[np.array], reversed=False) -> float:
    if reversed:
        max_n = max(len(tt_train_1), len(tt_train_2))
        long_tt = tt_train_1
        short_tt = tt_train_2
        if len(tt_train_1) < max_n:
            short_tt = tt_train_1
            long_tt = tt_train_2
        diff_n = len(long_tt) - len(short_tt)
        contraction_min_n = safe_multi_dot(
            [_tt_core_collapse(core_1, core_2) for core_1, core_2 in zip(long_tt[diff_n:], short_tt)])
        long_tt = long_tt[:diff_n - 1] + [np.einsum("rdl, lk -> rdk", long_tt[diff_n - 1], contraction_min_n)]
        return long_tt
    min_n = min(len(tt_train_1), len(tt_train_2))
    long_tt = tt_train_1[min_n:] if len(tt_train_1) > min_n else tt_train_2[min_n:]
    contraction_min_n = safe_multi_dot(
        [_tt_core_collapse(core_1, core_2) for core_1, core_2 in zip(tt_train_1[:min_n], tt_train_2[:min_n])])
    long_tt[0] = np.einsum("lr, rdk -> ldk", contraction_min_n, long_tt[0])
    return long_tt


def tt_to_tensor(tt_train):
    tensor = tt_train[0]
    for core in tt_train[1:]:
        tensor = np.tensordot(tensor, core, axes=(-1, 0))
    return np.sum(tensor, axis=(0, -1))


def _tt_influence_core_collapse(core, idx):
    return sum([
        np.kron(core[(slice(None),) + i], core[(slice(None),) + i])
        for i in product(*[list(range(1, int(idx + k == 0) - 1, -1)) for k in range(len(core.shape) - 2)])
    ])


def tt_influence(tt_train: List[np.array], idx):
    """
    Returns the influence of an idx-index atom on a boolean function
    """
    return np.sum(
        np.linalg.multi_dot([_tt_influence_core_collapse(core, idx - i) for i, core in enumerate(tt_train)]))


def _tt_shared_influence_core_collapse(core, idxs):
    return sum([
        np.kron(core[(slice(None),) + i], core[(slice(None),) + i])
        for i in product(*[list(range(1, int(np.max(idxs + k == 0)) - 1, -1)) for k in
                           range(len(core.shape) - 2)])
    ])


def tt_shared_influence(tt_train: List[np.array], idxs: np.array):
    """
    Returns the shared influence between an idx_1- and idx_2-index atom on a boolean function
    """
    return np.sum(
        np.linalg.multi_dot(
            [_tt_shared_influence_core_collapse(core, idxs - i) for i, core in enumerate(tt_train)]))


def _tt_phi_core(core: np.array):
    return sum([
        np.kron(
            np.expand_dims(core[(slice(None),) + i], list(range(1, 1 + len(i)))),
            phi(len(i) - 1)[(slice(None),) + sum(zip(i, [slice(None)] * len(i)), ())]
        ) for i in product(*([[0, 1]] * (len(core.shape) - 2)))])


def tt_walsh_op(tt_train: List[np.array]) -> List[np.array]:
    """
    Produces the -1, 1 binary tensor from a Fourier representation
    """
    return [_tt_phi_core(core) for core in tt_train]


def tt_pauli_op(tt_train: List[np.array], idx) -> List[np.array]:
    return tt_train[:idx] + [np.concatenate((tt_train[idx][:, None, 1], tt_train[idx][:, None, 0]), axis=1)] + tt_train[
                                                                                                               idx + 1:]


def _tt_phi_core_inv(core: np.array):
    return sum([
        np.kron(
            np.expand_dims(core[(slice(None),) + i], list(range(1, 1 + len(i)))),
            phi_inv(len(i) - 1)[(slice(None),) + sum(zip(i, [slice(None)] * len(i)), ())]
        ) for i in product(*([[0, 1]] * (len(core.shape) - 2)))])


def tt_walsh_op_inv(tt_train: List[np.array]) -> List[np.array]:
    """
    Produces the Fourier representation of a binary -1, 1 tensor
    """
    return [_tt_phi_core_inv(core) for core in tt_train]


def _tt_noisy_walsh_core(core: np.array, p_mat):
    return sum([
        np.kron(
            np.expand_dims(core[(slice(None),) + i], list(range(1, 1 + len(i)))),
            p_mat[(slice(None),) + sum(zip(i, [slice(None)] * len(i)), ())]
        ) for i in product(*([[0, 1]] * (len(core.shape) - 2)))])


def tt_noisy_walsh(tt_train: List[np.array], likelihoods: np.array):
    """
    Returns a formula weighted by a measure defined via the likelihoods
    """
    return [_tt_noisy_walsh_core(
        core,
        np.array([[1, 1], [p, -p]], dtype=float).reshape(1, 2, 2, 1)
    ) for core, p in zip(tt_train, likelihoods)]


def _tt_noisy_walsh_core_inv(core: np.array, p_mat):
    return sum([
        np.kron(
            np.expand_dims(core[(slice(None),) + i], list(range(1, 1 + len(i)))),
            p_mat[(slice(None),) + sum(zip(i, [slice(None)] * len(i)), ())]
        ) for i in product(*([[0, 1]] * (len(core.shape) - 2)))])


def tt_noisy_walsh_inv(tt_train: List[np.array], likelihoods: np.array):
    """
    Returns a formula weighted by a measure defined via the likelihoods
    """
    likelihoods = 1 / np.maximum(likelihoods, 1e-3)
    return [_tt_noisy_walsh_core_inv(
        core,
        (1 / 2) * np.array([[1, p], [1, -p]], dtype=float).reshape(1, 2, 2, 1)
    ) for core, p in zip(tt_train, likelihoods)]


def tt_noise_op(tt_train: List[np.array], likelihoods: np.array):
    return tt_walsh_op_inv(tt_noisy_walsh(tt_train, likelihoods))


def tt_noise_op_inv(tt_train: List[np.array], likelihoods: np.array):
    return tt_noisy_walsh_inv(tt_walsh_op(tt_train), likelihoods)


def tt_extract_seq(tt_train, assignments):
    N = len(tt_train)
    eps = 0.5 ** (1 / N) + 1 / 2 ** N
    tt_ttable = tt_rl_orthogonalise(tt_walsh_op(tt_train))
    answer = [np.array([1 - eps, eps]).reshape(1, 2, 1) for _ in range(N)]  # Sum over tensor sums to 1
    indices = list(range(N))
    for i in assignments.keys():
        a = assignments[i]
        answer[i] = ((a - 1) / (-2) * np.array([0.0, 1.0]) + (1 + a) / 2 * np.array([1.0, 0.0])).reshape(1, 2, 1)
        indices.remove(i)
    termination_crit = eps ** len(indices)  # if the 0.9s overlap on a truth value
    termination_crit -= 1 - termination_crit  # subtract all the other weights, i.e. 1 - true_entry (entries sum to 1)
    score = tt_inner_prod(tt_ttable, answer)
    while score < termination_crit:  # TODO: 1. If we let it run it will converge to the memory mean, 2. we can also after finding the minimal answer set flip that entry and proceed to run to get the next answer set
        improvements = -np.ones(N)
        buffer = deepcopy(answer)
        for j in indices:
            buffer[j] = np.array([eps, 1 - eps]).reshape(1, 2, 1)
            improvements[j] = tt_inner_prod(tt_ttable, buffer) - score
            buffer[j] = answer[j]
        max_improvements = np.argmax(improvements)
        answer[max_improvements] = np.array([1.0, 0.0]).reshape(1, 2, 1)
        indices.remove(max_improvements)
        score = tt_inner_prod(tt_ttable, answer)
    return [(np.array([1, -1]) @ np.round(a)).item() for a in answer]


def tt_xnor(tt_train_1: List[np.array], tt_train_2: List[np.array]) -> List[np.array]:
    """
    Produces the truth table result tensor
    """
    tt_train_xnor = tt_hadamard(tt_train_1, tt_train_2)
    return tt_rank_reduce(tt_train_xnor)


def tt_xor(tt_train_1: List[np.array], tt_train_2: List[np.array]) -> List[np.array]:
    return tt_xnor(tt_neg(tt_train_1), tt_train_2)


def tt_and(tt_train_1: List[np.array], tt_train_2: List[np.array]) -> List[np.array]:
    tt_train_1 = tt_scale(0.5, tt_train_1)
    tt_mul = tt_hadamard(tt_train_1, tt_train_2)
    tt_train_2 = tt_scale(0.5, tt_train_2)
    half = tt_scale(-0.5, tt_one(len(tt_train_1)))
    tt_train_and = tt_add(tt_add(half, tt_mul), tt_add(tt_train_1, tt_train_2))
    return tt_rank_reduce(tt_train_and)


def tt_or(tt_train_1: List[np.array], tt_train_2: List[np.array]) -> List[np.array]:
    return tt_neg(tt_and(tt_neg(tt_train_1), tt_neg(tt_train_2)))


def tt_neg(tt_train: List[np.array]) -> List[np.array]:
    return tt_scale(-1, tt_train)


def tt_scale(alpha, tt_train):
    idx = np.random.randint(low=0, high=len(tt_train))
    return sum([tt_train[:idx], [alpha * tt_train[idx]], tt_train[idx + 1:]], [])


def tt_normalise(tt_train, radius=1):
    return tt_scale(np.divide(radius, np.sqrt(tt_inner_prod(tt_train, tt_train))), tt_train)


def tt_abs(tt_train):
    rounded_tt_train = tt_binary_round(tt_train)
    absolute_tt_train = tt_rank_reduce(tt_hadamard(rounded_tt_train, tt_train))
    return absolute_tt_train


def tt_substitute(tt_train: List[np.array], substitutions: List[List[np.array]]) -> List[np.array]:
    """
    :param self:
    :param tt_train:
    :param substitutions: List of TTs to substitute into tt_train, substitutions must be in descending order of their
     indices in tt_train without gap in indices
    :return:
    """
    add_length = len(tt_train) - len(substitutions[0])
    assert add_length > 0, "There is no variable to substitute!"
    for tt_s in substitutions:
        add_length -= 1
        tt_core_without_basis = np.einsum("ldr, rk -> ldk", tt_train[-2], tt_train[-1][:, 0, :])
        tt_core_with_basis = np.einsum("ldr, rk -> ldk", tt_train[-2], tt_train[-1][:, 1, :])
        new_cores = tt_xnor(tt_train[:-2] + [tt_core_with_basis],
                            tt_s + [np.array([1, 0]).reshape(1, 2, 1)] * add_length)
        tt_train = tt_rank_reduce(tt_add(tt_train[:-2] + [tt_core_without_basis], new_cores))
    return tt_train


def tt_permute(tt_train: List[np.array], axes: List[Tuple]) -> List[np.array]:
    """
    Permutes a list of index tuples
    """
    for idx_1, idx_2 in axes:
        min_idx = min(idx_1, idx_2)
        max_idx = max(idx_1, idx_2)
        sub_half_core_min = np.zeros_like(tt_train[min_idx][:, None, 1, :])
        sub_half_core_max = np.zeros_like(tt_train[max_idx][:, None, 1, :])
        tt_min = tt_train[:min_idx] + [
            np.concatenate((tt_train[min_idx][:, None, 1, :], sub_half_core_min), axis=1)] + tt_train[
                                                                                             min_idx + 1:max_idx] + [
                     np.concatenate((tt_train[max_idx][:, None, 0, :], sub_half_core_max), axis=1)] + tt_train[
                                                                                                      max_idx + 1:]
        tt_max = tt_train[:min_idx] + [
            np.concatenate((tt_train[min_idx][:, None, 0, :], sub_half_core_min), axis=1)] + tt_train[
                                                                                             min_idx + 1:max_idx] + [
                     np.concatenate((tt_train[max_idx][:, None, 1, :], sub_half_core_max), axis=1)] + tt_train[
                                                                                                      max_idx + 1:]
        tt_basis_1 = tt_train[:min_idx] + [
            np.concatenate((tt_train[min_idx][:, None, 0, :], sub_half_core_min), axis=1)] + tt_train[
                                                                                             min_idx + 1:max_idx] + [
                         np.concatenate((tt_train[max_idx][:, None, 0, :], sub_half_core_max), axis=1)] + tt_train[
                                                                                                          max_idx + 1:]
        tt_basis_2 = tt_train[:min_idx] + [
            np.concatenate((sub_half_core_min, tt_train[min_idx][:, None, 1, :]), axis=1)] + tt_train[
                                                                                             min_idx + 1:max_idx] + [
                         np.concatenate((sub_half_core_max, tt_train[max_idx][:, None, 1, :]), axis=1)] + tt_train[
                                                                                                          max_idx + 1:]
        tt_min = tt_xnor(tt_min, tt_atom_train(max_idx, len(tt_min)))
        tt_max = tt_xnor(tt_max, tt_atom_train(min_idx, len(tt_max)))
        tt_train = tt_rank_reduce(tt_add(tt_add(tt_basis_1, tt_min), tt_add(tt_basis_2, tt_max)))
    return tt_train


def tt_add_column(tt_block_train, tt_column, i):
    binary_i = bin(i)[2:]
    binary_i = '0' * (len(tt_block_train) - len(binary_i)) + binary_i
    index = [(np.expand_dims(np.zeros_like(c), 1), np.expand_dims(c, 1)) if bool(int(b)) else (
        np.expand_dims(c, 1), np.expand_dims(np.zeros_like(c), 1)) for
             b, c in zip(binary_i, tt_column)]
    tt_column = [np.concatenate(tup, axis=1) for tup in index]
    return tt_add(tt_block_train, tt_column)


def tt_linear_op_from_columns(tt_trains: List[np.array]):
    n = len(tt_trains)
    bin_length = len(bin(n - 1)[2:])
    binary_i = '0' * bin_length
    index = [(np.expand_dims(np.zeros_like(c), 1), np.expand_dims(c, 1)) if bool(int(b)) else (
        np.expand_dims(c, 1), np.expand_dims(np.zeros_like(c), 1)) for b, c in
             zip(binary_i, tt_trains[0][-bin_length:])]
    tt_block_tensor = tt_trains[0][:-bin_length] + [np.concatenate(tup, axis=1) for tup in index]
    for i in range(1, n):
        tt_block_tensor = tt_add_column(tt_block_tensor, tt_trains[i])
        tt_block_tensor = tt_rank_reduce(tt_block_tensor)

    return tt_block_tensor, bin_length


def tt_conjugate_gradient(linear_op: List[np.array], tt_train: List[np.array], num_iter=10, tol=1e-7):
    x = [np.zeros((1, 2, 1)) for _ in range(len(tt_train))]
    r = p = tt_train
    r_2 = tt_inner_prod(r, r)
    for _ in range(num_iter):
        prev_r_2 = r_2
        l_op_p = tt_linear_op(linear_op, p)
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


def tt_max_eigentensor(linear_op: List[np.array], num_iter=10, tol=1e-3):
    normalisation = np.sqrt(tt_inner_prod(linear_op, linear_op))
    linear_op = tt_scale(np.divide(1, normalisation), linear_op)
    eig_vec = tt_normalise([np.random.randn(1, 2, 1) for _ in range(len(linear_op))])
    linear_op = tt_rank_reduce(linear_op)
    norm_2 = np.inf
    for i in range(num_iter):
        prev_norm_2 = norm_2
        eig_vec = tt_linear_op(linear_op, eig_vec)
        norm_2 = tt_inner_prod(eig_vec, eig_vec)
        eig_vec = tt_scale(np.divide(1, np.sqrt(norm_2)), eig_vec)
        if np.less_equal(np.abs(norm_2 - prev_norm_2), tol):
            break
    prev_eig_vec = eig_vec
    eig_vec = tt_linear_op(linear_op, eig_vec)
    eig_val = tt_inner_prod(prev_eig_vec, eig_vec)
    return tt_normalise(eig_vec), normalisation * eig_val


def _tt_op_core_collapse(linear_core_op: np.array, core_2: np.array) -> np.array:
    indices_op = (slice(None),) * (len(linear_core_op.shape) - 2)
    indices_core = (slice(None),) + (None,) * (len(linear_core_op.shape) - len(core_2.shape))
    return sum([
        np.kron(linear_core_op[indices_op + (i,)], core_2[indices_core + (i,)])
        for i in range(core_2.shape[1])
    ])


def _tt_linear_op(linear_op: List[np.array], tt_train: List[np.array]) -> List[np.array]:
    split_idx = np.argmax([len(c.shape) for c in linear_op])
    op_length = len(linear_op) - split_idx
    full_cores = [_tt_op_core_collapse(core_op, core) for core_op, core in
                  zip(linear_op[split_idx:], tt_train[-op_length:])]
    left_overs = len(tt_train) - op_length
    if left_overs > 0:
        half_core = safe_multi_dot(
            [_tt_core_collapse(core_op, core) for core_op, core in
             zip(linear_op[split_idx - left_overs:split_idx], tt_train[:-op_length])]
        )
        full_cores[0] = np.einsum("ab, bce -> ace", half_core, full_cores[0])
    full_cores = linear_op[:split_idx - left_overs] + full_cores
    return full_cores


def tt_linear_op(linear_op, tt_train: List[np.array], bound=None) -> List[np.array]:
    tt_train = _tt_linear_op(linear_op, tt_train)
    return tt_rank_reduce(tt_train, bound)


def tt_randomised_linear_op(linear_op, tt_train: List[np.array], ranks) -> List[np.array]:
    return tt_lr_random_orthogonalise(_tt_linear_op(linear_op, tt_train), target_ranks=ranks)


def _tt_op_op_collapse(linear_core_op_1, linear_core_op_2):
    return sum([
        np.kron(linear_core_op_1[:, None, i], linear_core_op_2[:, :, None, i])
        for i in range(linear_core_op_2.shape[2])
    ])


def tt_linear_op_compose(linear_op_1, linear_op_2):
    split_idx = np.argmax([len(c.shape) for c in linear_op_1])
    half_core = safe_multi_dot([
        _tt_core_collapse(core_op_1, core_op_2) for core_op_1, core_op_2 in
        zip(linear_op_1[:split_idx], linear_op_2[:split_idx])
    ])
    full_cores = [
        _tt_op_op_collapse(core_op_1, core_op_2) for core_op_1, core_op_2 in
        zip(linear_op_1[split_idx:], linear_op_2[split_idx:])
    ]
    if len(half_core) > 0:
        full_cores[0] = np.einsum("ab, bcde -> acde", half_core, full_cores[0])
    return full_cores


def tt_transpose(linear_op):
    """
    Constructs the transpose of the linear op
    """
    split_idx = np.argmax([len(c.shape) for c in linear_op])
    return linear_op[:split_idx] + [np.swapaxes(c, axis1=1, axis2=2) for c in linear_op[split_idx:]]


def tt_gram(linear_op):
    """ Constructs the gram tensor for a linear op"""
    tt_linear_op_t = tt_transpose(linear_op)
    gram = tt_linear_op_compose(linear_op, tt_linear_op_t)
    gram = tt_rank_reduce(gram)
    return gram


def tt_trace(linear_op):
    I = tt_identity(len(linear_op))
    return tt_inner_prod(linear_op, I)


def tt_min_eigentensor(linear_op: List[np.array], num_iter=10, tol=1e-3):
    """
    Only for symmetric matrices
    """
    n = len(linear_op)
    normalisation = np.sqrt(tt_inner_prod(linear_op, linear_op))
    linear_op = tt_scale(-np.divide(1, normalisation), linear_op)
    identity = tt_identity(n)
    identity = tt_scale(2, identity)
    linear_op = tt_add(identity, linear_op)
    linear_op = tt_rank_reduce(linear_op, tt_bound=0)
    eig_vec = tt_normalise([np.random.randn(1, 2, 1) for _ in range(n)])
    norm_2 = np.inf
    for i in range(num_iter):
        prev_norm_2 = norm_2
        eig_vec = tt_linear_op(linear_op, eig_vec, bound=0)
        norm_2 = tt_inner_prod(eig_vec, eig_vec)
        eig_vec = tt_scale(np.divide(1, np.sqrt(norm_2)), eig_vec)
        if np.less_equal(np.abs(norm_2 - prev_norm_2), tol):
            break
    prev_eig_vec = eig_vec
    eig_vec = tt_linear_op(linear_op, eig_vec, bound=0)
    eig_val = tt_inner_prod(prev_eig_vec, eig_vec)
    return tt_normalise(eig_vec), normalisation * (2 - eig_val)


def tt_randomised_min_eigentensor(linear_op: List[np.array], num_iter=10, tol=1e-3):
    ranks = [2 ** i for i in range(int(np.ceil((len(linear_op) + 1) / 2)))]
    target_ranks = ranks + list(reversed(ranks[:len(linear_op) + 1 - len(ranks)]))
    tt_gaussian = [
        np.divide(1, l_n * 2 * l_np1) * np.random.randn(l_n, 2, l_np1)
        for i, (l_n, l_np1) in enumerate(zip(target_ranks[:-1], target_ranks[1:]))
    ]
    return _tt_randomised_min_eigentensor(linear_op, tt_gaussian, num_iter, tol)


def _tt_randomised_min_eigentensor(linear_op: List[np.array], tt_gaussian, num_iter=10, tol=1e-3):
    """
    Only for symmetric matrices
    """
    n = len(linear_op)
    normalisation = np.sqrt(tt_inner_prod(linear_op, linear_op))
    linear_op = tt_scale(-np.divide(1, normalisation), linear_op)
    identity = tt_identity(n)
    identity = tt_scale(2, identity)
    linear_op = tt_rank_reduce(tt_add(identity, linear_op), tt_bound=tol)
    eig_vec = tt_normalise([np.random.randn(1, 2, 1) for _ in range(n)])
    norm_2 = np.inf
    for i in range(num_iter):
        prev_norm_2 = norm_2
        eig_vec = _tt_lr_random_orthogonalise(_tt_linear_op(linear_op, eig_vec), tt_gaussian)
        norm_2 = tt_inner_prod(eig_vec, eig_vec)
        eig_vec = tt_scale(np.divide(1, np.sqrt(norm_2)), eig_vec)
        if np.less_equal(np.abs(norm_2 - prev_norm_2), tol):
            break
    prev_eig_vec = eig_vec
    eig_vec = _tt_lr_random_orthogonalise(_tt_linear_op(linear_op, eig_vec), tt_gaussian)
    eig_val = tt_inner_prod(prev_eig_vec, eig_vec)
    return tt_normalise(eig_vec), normalisation * (2 - eig_val)


def _trivial_step_size(it):
    return np.minimum(1.0, 2.0 / it)


def _tt_op_vec_collapse(linear_core_op: np.array, core_2: np.array) -> np.array:
    return sum([
        np.kron(linear_core_op[:, i], core_2[:, i, None, None])
        for i in range(core_2.shape[1])
    ])


def tt_constraint_contract(linear_op, lag_mul):
    split_idx = np.argmax([len(c.shape) for c in linear_op])
    return linear_op[:split_idx] + [_tt_op_vec_collapse(op, l) for op, l in zip(linear_op[split_idx:], lag_mul)]


def tt_outer_product(tt_train_1, tt_train_2):
    outer = [np.kron(np.expand_dims(c_1, 1), np.expand_dims(c_2, 2)) for c_1, c_2 in zip(tt_train_1, tt_train_2)]
    return outer


def tt_binary_round(tt_train, num_iter=30, tol=1e-10):
    for _ in range(num_iter):
        tt_train_p2 = tt_rank_reduce(tt_hadamard(tt_train, tt_train))
        boolean_poly = tt_walsh_op_inv(tt_train_p2)
        if np.less_equal(np.abs(tt_inner_prod(boolean_poly, boolean_poly) - 1), tol):
            break
        tt_train_p3 = tt_rank_reduce(tt_hadamard(tt_train_p2, tt_train))
        tt_update = tt_scale(-0.5, tt_train_p3)
        next_tt_train = tt_scale(1.5, tt_train)
        tt_train = tt_rank_reduce(tt_add(next_tt_train, tt_update))
    return tt_train


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
                          tt_bound=0)


def tt_op_to_matrix(linear_op):
    tensor = tt_to_tensor(linear_op)
    n = len(tensor.shape)
    axes = [i for i in range(1, n, 2)] + [i for i in range(0, n - 1, 2)]
    return np.transpose(tensor, axes).reshape(np.prod(tensor.shape[:n // 2]), -1)


def _cg_oracle(tt_eig_sketch, lag_mul_1, lag_mul_2, obj_sdp, linear_op_sdp, res, trace_param_root_n, tol):
    lag = tt_add(lag_mul_1, tt_scale(lag_mul_2, res))
    constraint_term = tt_constraint_contract(linear_op_sdp, lag)
    sdp_gradient = tt_add(obj_sdp, constraint_term)
    tt_eig, min_eig_val = _tt_randomised_min_eigentensor(sdp_gradient, tt_eig_sketch, num_iter=2000, tol=tol)
    current_trace_param = trace_param_root_n[0] if min_eig_val > 0 else trace_param_root_n[1]
    tt_eig = [np.sqrt(current_trace_param) * c for c in tt_eig]
    return sdp_gradient, tt_eig, lag, np.power(current_trace_param, len(tt_eig_sketch)) * min_eig_val


def _tt_ops_ops_collapse(linear_core_op_1, linear_core_op_2):
    return sum([
        np.kron(linear_core_op_1[:, None, :, i, j], linear_core_op_2[:, :, None, i, j])
        for (i, j) in product([0, 1], [0, 1])
    ])


def _tt_custom_collapse(core_1, core_2):
    return sum([
        _tt_train_kron(core_1, core_2)[:, i]
        for i in [0, 1]
    ])


def _interpolate(obj_sdp, linear_ops, sdp_gradient, gamma, objective, z_constraints, X, tt_eig, idx):
    tt_Eig = [np.concatenate((np.expand_dims(c, axis=2), np.expand_dims(c, axis=2)), axis=2) for c in tt_eig]
    print(np.round(tt_op_to_matrix(tt_Eig), decimals=2))
    print(np.round(tt_op_to_matrix(X), decimals=2))
    d = tt_sub(tt_Eig, X)
    print(np.round(tt_op_to_matrix(d), decimals=2))
    # TODO: Need each column to be separated into an op, then take eval_constraints(d_op, sdp_gradient)
    d_op = [np.concatenate((np.expand_dims(c, axis=3), np.expand_dims(np.zeros_like(c), axis=3)), axis=3) for c in d]
    sdp_gradient_op = [np.concatenate((np.expand_dims(c, axis=3), np.expand_dims(np.zeros_like(c), axis=3)), axis=3) for
                       c in sdp_gradient]
    #print(np.round(tt_op_to_matrix([d_op[0][:, 1]] + [c[:, 0] for c in d_op[1:]]), decimals=2))
    #print([c.shape for c in d_op])
    #print(np.round(tt_to_tensor(tt_eval_constraints(d_op, tt_transpose([np.array([[1, 1], [0, 0]]).reshape(1, 2, 2, 1) for _ in range(len(d_op))]))), decimals=2))
    #print("Eigs", np.linalg.eigvals(tt_op_to_matrix(tt_linear_op_compose(d, tt_transpose(d)))))
    #print(tt_op_to_matrix(tt_linear_op_compose(d, tt_transpose(d))))
    #print(np.round(tt_to_tensor(tt_eval_constraints(sdp_gradient_op, d)), decimals=2))
    #print([c.shape for c in sdp_gradient_op])
    #print([c.shape for c in d])
    print(np.round(tt_op_to_matrix(sdp_gradient), decimals=2))
    print(tt_op_to_matrix(d)[:, -4] @ tt_op_to_matrix(sdp_gradient) @ tt_op_to_matrix(d)[:, -4])
    print(tt_op_to_matrix(d)[:, -3] @ tt_op_to_matrix(sdp_gradient) @ tt_op_to_matrix(d)[:, -3])
    print(tt_op_to_matrix(d)[:, -2] @ tt_op_to_matrix(sdp_gradient) @ tt_op_to_matrix(d)[:, -2])
    print(tt_op_to_matrix(d)[:, -1] @ tt_op_to_matrix(sdp_gradient) @ tt_op_to_matrix(d)[:, -1])
    #print(np.round(tt_op_to_matrix([c[:, 1] for c in sdp_gradient_op]), decimals=2))
    #sim_d = tt_eval_constraints(d_op, [c[:, 1] for c in sdp_gradient_op])
    sim_d = [_tt_ops_ops_collapse(d_core, sdp_core) for d_core, sdp_core in zip(d_op, sdp_gradient_op)]
    print(np.round(tt_op_to_matrix(sim_d), decimals=2))
    print([c.shape for c in d])
    print([c.shape for c in sim_d])
    sim = [_tt_custom_collapse(c_1, c_2) for c_1, c_2 in zip(tt_transpose(d), sim_d)]
    print([c.shape for c in sim])
    print(np.round(tt_to_tensor(sim), decimals=2))
    #max_index = tt_argmax(sim)

    scaled_tt_eig = tt_scale(np.sqrt(gamma), tt_eig)
    X = tt_scale(np.sqrt(1 - gamma), X)
    idx = idx % (2 ** len(scaled_tt_eig))
    X = tt_add_column(X, scaled_tt_eig, idx)
    X = tt_rank_reduce(X, tt_bound=1e-10)  # _tt_lr_random_orthogonalise(X, tt_X_sketch)
    constraint_update = tt_linear_op(tt_linear_op(linear_ops, tt_eig), tt_eig)
    z_constraints = tt_rank_reduce(tt_add(tt_scale(1 - gamma, z_constraints), tt_scale(gamma, constraint_update)),
                                   tt_bound=1e-10)
    objective = (1 - gamma) * objective + gamma * tt_inner_prod(tt_eig, tt_linear_op(obj_sdp, tt_eig))
    return objective, z_constraints, X


def _tt_ops_core_collapse(linear_ops_core: np.array, core_2: np.array) -> np.array:
    return sum([
        np.kron(linear_ops_core[:, :, i, j], core_2[:, None, i, j])
        for (i, j) in product([0, 1], [0, 1])
    ])


def tt_eval_constraints(linear_ops, X):
    split_idx = np.argmax([len(c.shape) for c in linear_ops])
    factor = safe_multi_dot(
        [_tt_core_collapse(core_1, core_2) for core_1, core_2 in zip(linear_ops[:split_idx], X[:split_idx])])
    full_cores = [_tt_ops_core_collapse(core_1, core_2) for core_1, core_2 in
                  zip(linear_ops[split_idx:], X[split_idx:])]
    if len(factor) > 0:
        full_cores[0] = np.einsum("ab, bcd -> acd", factor, full_cores[0])
    return full_cores


def tt_sketch_like(tt_train, target_ranks):
    return [
        np.divide(1, l_n * np.prod(tt_train[i].shape[1:-1]) * l_np1) * np.random.randn(l_n, *tt_train[i].shape[1:-1],
                                                                                       l_np1)
        for i, (l_n, l_np1) in enumerate(zip(target_ranks[:-1], target_ranks[1:]))
    ]


def tt_sketch(shape, target_ranks):
    return [
        np.divide(1, l_n * np.prod(shape) * l_np1) * np.random.randn(l_n, *shape, l_np1)
        for i, (l_n, l_np1) in enumerate(zip(target_ranks[:-1], target_ranks[1:]))
    ]


def tt_sdp_fw(
    obj_sdp: List[np.array],
    linear_ops: List[np.array],
    bias: List[np.array],
    trace_param_root_n,
    dual_gap_tol=1e-6,
    num_iter=100
):
    tol = np.divide(1, np.power(2, 2 * len(obj_sdp) + 2))
    X = [np.zeros((1, 2, 2, 1)) for _ in range(len(linear_ops))]
    neg_bias = tt_scale(-1, bias)
    res = neg_bias
    lag_mul_1 = [np.zeros((1, *b.shape[1:-1], 1)) for b in bias]
    z_constraints = copy.copy(lag_mul_1)
    lag_mul_2 = 1
    alpha_0 = 4 * tt_inner_prod(linear_ops, linear_ops) * trace_param_root_n[1] ** (2 * len(X))
    duality_gaps = []
    target_ranks = [1] + list(np.maximum(tt_ranks(obj_sdp), tt_ranks(linear_ops))) + [1]
    tt_X_sketch = tt_sketch_like(obj_sdp, target_ranks)
    ranks = [2 ** i for i in range(int(np.ceil((len(lag_mul_1) + 1) / 2)))]
    lag_target_ranks = ranks + list(reversed(ranks[:len(lag_mul_1) + 1 - len(ranks)]))
    tt_lag_sketch = tt_sketch_like(lag_mul_1, lag_target_ranks)
    tt_eig_sketch = tt_sketch((2,), lag_target_ranks)
    it = 1
    objective = 0
    for it in range(1, num_iter):
        sdp_gradient, tt_eig, lag, eig_val = _cg_oracle(
            tt_eig_sketch, lag_mul_1, lag_mul_2, obj_sdp, linear_ops, res, trace_param_root_n, tol)
        duality_gap = objective + tt_inner_prod(lag, z_constraints) - eig_val
        duality_gaps.append(duality_gap)
        if np.less_equal(np.abs(duality_gap), dual_gap_tol):
            break
        gamma = _trivial_step_size(it)
        objective, z_constraints, X = _interpolate(
            obj_sdp, linear_ops, sdp_gradient, gamma, objective, z_constraints, X, tt_eig, it - 1
        )
        res = tt_add(z_constraints,
                     neg_bias)  #_tt_lr_random_orthogonalise(tt_add(tt_eval_constraints(linear_ops, X), neg_bias), tt_lag_sketch)
        print("Solution ranks: ", tt_ranks(X))
        print("Constraint error: ", tt_inner_prod(res, res), "Objective: ", objective)
        alpha = min(np.divide(alpha_0, np.power(it + 1, 3 / 2) * tt_inner_prod(res, res)), 1)
        lag_mul_1 = tt_rank_reduce(tt_add(lag_mul_1, tt_scale(alpha, res)),
                                   tt_bound=1e-10)  #_tt_lr_random_orthogonalise(tt_add(lag_mul_1, tt_scale(alpha, res)), tt_lag_sketch)
        lag_mul_2 = np.sqrt(it + 1)
    print(f"Finished after {it} iterations")
    X = tt_linear_op_compose(X, tt_transpose(X))
    return tt_rank_reduce(X), duality_gaps


def _core_mask(core, i, j):
    mask = np.zeros_like(core)
    mask[:, i, j] += core[:, i, j]
    return np.expand_dims(mask, 1)


def tt_mask_to_linear_op(tt_train):
    return tt_rank_reduce(
        [np.concatenate([_core_mask(c, i, j) for (i, j) in product([0, 1], [0, 1])], axis=1) for c in tt_train]
    )


def tt_argmax(tt_train, p=1):
    orth_tt_train = tt_rl_orthogonalise(copy.copy(tt_train))
    k = 2 * (len(orth_tt_train))
    d = len(orth_tt_train)
    p0 = p / d  # Scale factor (2^p0) for each TT-core

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


def _als_grad_22(A, C, X):
    """
    gradient for A(C kron X)
    """
    m, n = C.shape
    p, q = X.shape
    I_nq = np.eye(n * q)
    I_n = np.eye(n)
    I_q = np.eye(q)
    I_p = np.eye(p)
    K_qm = commutation_matrix(q, m)
    H = np.kron(I_n, K_qm) @ np.kron(vec(C), I_q)
    L = np.kron(I_nq, A) @ np.kron(H, I_p)
    return mat(vec(I_nq).T @ L, X.shape)


def _als_grad_33(A, X, C):
    p, q = C.shape
    m, n = X.shape
    I_nq = np.eye(n * q)
    I_n = np.eye(n)
    I_m = np.eye(m)
    I_p = np.eye(p)
    K_qm = commutation_matrix(q, m)
    G = np.kron(K_qm, I_p) @ np.kron(I_m, vec(C))
    L = np.kron(I_nq, A) @ np.kron(I_n, G)
    return mat(vec(I_nq).T @ L, X.shape)


def _als_grad_44(A, X):
    m, n = X.shape
    I_n = np.eye(n)
    I_m = np.eye(m)
    K_nn = commutation_matrix(n, n)
    K_mm = commutation_matrix(m, m)
    H = np.kron(I_n, K_nn) @ np.kron(vec(I_n), I_n)
    G = np.kron(K_mm, I_m) @ np.kron(I_m, vec(I_m))
    D = np.kron(H.T, I_m) @ np.kron(np.kron(I_n, A), I_m) @ np.kron(I_n, G)
    return mat(D @ vec(X) + D.T @ vec(X), X.shape)


def _tt_burer_monteiro_grad(A_22, A_33, A_44, C_00, C_01, C_10, C_11, V_00, V_01, V_10, V_11):
    m, n = C_00.shape
    orig_p, orig_q = V_00.shape
    p = orig_p ** 2
    q = orig_q ** 2
    I = np.eye(max(q ** 2, p ** 2))
    K_np = I[np.arange(n * p).reshape((n, p), order="F").T.ravel(order="F"), :n * p]
    K_qm = I[np.arange(q * m).reshape((q, m), order="F").T.ravel(order="F"), :q * m]
    K_orig_qp = I[np.arange(orig_q * orig_p).reshape((orig_q, orig_p), order="F").T.ravel(order="F"), :orig_q * orig_p]
    H_V_00 = np.kron(I[:orig_q, :orig_q], K_orig_qp) @ np.kron(vec(V_00), I[:orig_q, :orig_q])
    H_V_01 = np.kron(I[:orig_q, :orig_q], K_orig_qp) @ np.kron(vec(V_01), I[:orig_q, :orig_q])
    G_V_00 = np.kron(K_orig_qp, I[:orig_p, :orig_p]) @ np.kron(I[:orig_p, :orig_p], vec(V_00))
    G_V_01 = np.kron(K_orig_qp, I[:orig_p, :orig_p]) @ np.kron(I[:orig_p, :orig_p], vec(V_01))
    S_22 = A_22.T.reshape(p, m * n * q, order="F") @ np.kron(I[:n, :n], K_qm)
    S_33 = np.kron(K_np.T, I[:m, :m]) @ A_33.T.reshape(m * n * p, q, order="F")
    L_22C_00 = S_22 @ np.kron(vec(C_00), I[:q, :q])
    L_22C_01 = S_22 @ np.kron(vec(C_01), I[:q, :q])
    L_22C_10 = S_22 @ np.kron(vec(C_10), I[:q, :q])
    L_33C_00 = np.kron(I[:p, :p], vec(C_00).T) @ S_33
    L_33C_01 = np.kron(I[:p, :p], vec(C_01).T) @ S_33
    L_33C_10 = np.kron(I[:p, :p], vec(C_10).T) @ S_33

    H_44 = I[:q ** 2, :q ** 2].reshape(q ** 3, q, order="F")
    G_44 = I[:p ** 2, :p ** 2].reshape(p ** 3, p, order="F")

    S_44_1 = H_44.T @ np.kron(I[:q, :q], A_44)
    S_44_2 = np.kron(A_44, I[:p, :p]) @ G_44

    pair_1 = np.kron(V_00, V_00) + np.kron(V_10, V_10)
    pair_2 = np.kron(V_01, V_00) + np.kron(V_11, V_10)
    pair_3 = np.kron(V_00, V_01) + np.kron(V_10, V_11)

    D_1 = vec(G_44.T @ (pair_1 @ S_44_1).reshape(p ** 3, q, order="F") + (S_44_2 @ pair_1).reshape(p, q ** 3,
                                                                                                   order="F") @ H_44).T
    D_2 = vec(G_44.T @ (pair_2 @ S_44_1).reshape(p ** 3, q, order="F") + (S_44_2 @ pair_2).reshape(p, q ** 3,
                                                                                                   order="F") @ H_44).T
    D_3 = vec(G_44.T @ (pair_3 @ S_44_1).reshape(p ** 3, q, order="F") + (S_44_2 @ pair_3).reshape(p, q ** 3,
                                                                                                   order="F") @ H_44).T
    L_1 = D_1 @ (np.kron(H_V_00, I[:orig_p, :orig_p]) + np.kron(I[:orig_q, :orig_q], G_V_00))
    L_2 = D_2 @ np.kron(H_V_01, I[:orig_p, :orig_p])
    L_3 = D_3 @ np.kron(I[:orig_q, :orig_q], G_V_01)

    L = mat(L_1 + L_2 + L_3, V_00.shape)

    return L + (
        G_V_00.T @ (L_22C_00 + L_33C_00).reshape(-1, orig_q, order="F")
        + (L_22C_00 + L_33C_00).reshape(orig_p, -1, order="F") @ H_V_00
        + (L_22C_01 + L_33C_01).reshape(orig_p, -1, order="F") @ H_V_01
        + G_V_01.T @ (L_22C_10 + L_33C_10).reshape(-1, orig_q, order="F")
    )


def _tt_als_core_wise(tt_train, cores, idx, lr=0.1):
    r_i, d, r_ip1 = tt_train[idx].shape
    diff = tt_add(tt_train, cores)  # the order matters here
    prev_error = tt_inner_prod(diff, diff)
    if idx == 0:
        outer_contraction = np.diag(
            safe_multi_dot([_tt_core_collapse(core, core) for core in diff[idx + 1:]]).flatten())
        X_0 = np.diag(cores[idx][:, 0].flatten())
        C_0 = np.diag(tt_train[idx][:, 0].flatten())
        X_1 = np.diag(cores[idx][:, 1].flatten())
        C_1 = np.diag(tt_train[idx][:, 1].flatten())
    elif idx == len(cores) - 1:
        outer_contraction = np.diag(
            safe_multi_dot([_tt_core_collapse(core, core) for core in diff[:idx]]).flatten())
        X_0 = np.diag(cores[idx][:, 0].flatten())
        C_0 = np.diag(tt_train[idx][:, 0].flatten())
        X_1 = np.diag(cores[idx][:, 1].flatten())
        C_1 = np.diag(tt_train[idx][:, 1].flatten())
    else:
        left_contraction = safe_multi_dot([_tt_core_collapse(core, core) for core in diff[:idx]]).reshape(1, -1)
        right_contraction = safe_multi_dot([_tt_core_collapse(core, core) for core in diff[idx + 1:]]).reshape(-1, 1)
        outer_contraction = right_contraction @ left_contraction
        X_0 = cores[idx][:, 0]
        C_0 = tt_train[idx][:, 0]
        X_1 = cores[idx][:, 1]
        C_1 = tt_train[idx][:, 1]
    s_1 = np.array(
        [[i + X_0.shape[1] * j for i in range(X_0.shape[1])] for j in np.arange(0, 2 * X_0.shape[1], 2)]).flatten()
    s_2 = np.array(
        [[i + X_0.shape[0] * j for i in range(X_0.shape[0])] for j in np.arange(0, 2 * X_0.shape[0], 2)]).flatten()
    block_diag_1 = outer_contraction[:outer_contraction.shape[0] // 2, :outer_contraction.shape[1] // 2]
    block_diag_2 = outer_contraction[outer_contraction.shape[0] // 2:, outer_contraction.shape[1] // 2:]
    A_22 = block_diag_1[np.ix_(s_1 + X_0.shape[1], s_2 + X_0.shape[0])]
    A_33 = block_diag_2[np.ix_(s_1, s_2)]
    A_44 = block_diag_2[np.ix_(s_1 + X_0.shape[1], s_2 + X_0.shape[0])]
    vec_0 = _als_grad_22(A_22, C_0, X_0) + _als_grad_33(A_33, X_0, C_0) + _als_grad_44(A_44, X_0)
    vec_1 = _als_grad_22(A_22, C_1, X_1) + _als_grad_33(A_33, X_1, C_1) + _als_grad_44(A_44, X_1)
    if idx == 0:
        cores[idx][:, 0, :] -= lr * np.expand_dims(np.diagonal(vec_0.reshape(r_ip1, r_ip1)), axis=0)
        cores[idx][:, 1, :] -= lr * np.expand_dims(np.diagonal(vec_1.reshape(r_ip1, r_ip1)), axis=0)
    elif idx == len(cores) - 1:
        cores[idx][:, 0, :] -= lr * np.expand_dims(np.diagonal(vec_0.reshape(r_i, r_i)), axis=-1)
        cores[idx][:, 1, :] -= lr * np.expand_dims(np.diagonal(vec_1.reshape(r_i, r_i)), axis=-1)
    else:
        cores[idx][:, 0, :] -= lr * vec_0.reshape(r_i, r_ip1)
        cores[idx][:, 1, :] -= lr * vec_1.reshape(r_i, r_ip1)
    diff = tt_add(tt_train, cores)  # the order matters here
    error = tt_inner_prod(diff, diff)
    lr = min(0.999 * (prev_error / error) * lr, 0.5)
    print(f"Error: {error}, {lr}")
    return cores, lr


def tt_als(tt_train, max_iter=100):
    tt_train = tt_rank_reduce([-1 * tt_train[0]] + tt_train[1:])
    ranks = [1] + tt_ranks(tt_train) + [1]
    cores = [np.divide(1, r_i * 2 * r_ip1) * np.random.randn(r_i, 2, r_ip1) for r_i, r_ip1 in
             zip(ranks[:-1], ranks[1:])]
    indices = list(range(len(cores))) + list(reversed(range(len(cores))))
    lr = 0.5 * np.ones(len(cores))
    for iteration in range(max_iter):
        for k in indices:
            print(f"Core number {k}")
            cores, l = _tt_als_core_wise(tt_train, cores, k, lr=lr[k])
            lr[k] = l

    return cores


def func(tt_train, cores):
    comp = tt_linear_op_compose(cores, tt_transpose(cores))
    diff = tt_add(tt_train, comp)
    return tt_inner_prod(diff, diff)


def _tt_bm_core_wise(tt_train, cores, idx, prev_error, lr=0.1, tol=1e-3):
    xr_i, _, _, xr_ip1 = cores[idx].shape
    comp = tt_linear_op_compose(cores, tt_transpose(cores))
    diff = tt_add(tt_train, comp)
    error = tt_inner_prod(diff, diff)
    lr = min(0.999 * (prev_error / error) * lr, 0.5)
    print(f"Error: {error}, {lr}")
    if np.less_equal(error, tol):
        return cores
    elif 0 < idx < len(cores) - 1:
        left_contraction = safe_multi_dot([_tt_core_collapse(core, core) for core in diff[:idx]]).reshape(1, -1)
        right_contraction = safe_multi_dot([_tt_core_collapse(core, core) for core in diff[idx + 1:]]).reshape(-1, 1)
        outer_contraction = right_contraction @ left_contraction
        V_00 = cores[idx][:, 0, 0]
        C_00 = tt_train[idx][:, 0, 0]
        V_01 = cores[idx][:, 0, 1]
        C_01 = tt_train[idx][:, 0, 1]
        V_10 = cores[idx][:, 1, 0]
        C_10 = tt_train[idx][:, 1, 0]
        V_11 = cores[idx][:, 1, 1]
        C_11 = tt_train[idx][:, 1, 1]
    else:
        if idx == 0:
            outer_contraction = np.diag(
                safe_multi_dot([_tt_core_collapse(core, core) for core in diff[idx + 1:]]).flatten())
        else:
            outer_contraction = np.diag(
                safe_multi_dot([_tt_core_collapse(core, core) for core in diff[:idx]]).flatten())
        V_00 = np.diag(cores[idx][:, 0, 0].flatten())
        C_00 = np.diag(tt_train[idx][:, 0, 0].flatten())
        V_10 = np.diag(cores[idx][:, 1, 0].flatten())
        C_10 = np.diag(tt_train[idx][:, 1, 0].flatten())
        V_01 = np.diag(cores[idx][:, 0, 1].flatten())
        C_01 = np.diag(tt_train[idx][:, 0, 1].flatten())
        V_11 = np.diag(cores[idx][:, 1, 1].flatten())
        C_11 = np.diag(tt_train[idx][:, 1, 1].flatten())

    m, n = C_00.shape
    p, q = V_00.shape
    p *= p
    q *= q

    cv_y = np.array([[i + (m + p) * j for i in range(p)] for j in range(m)]).flatten()
    cv_x = np.array([[i + (n + q) * j for i in range(q)] for j in range(n)]).flatten()

    vc_y = np.array([[i + (m + p) * j for i in range(m)] for j in range(p)]).flatten()
    vc_x = np.array([[i + (n + q) * j for i in range(n)] for j in range(q)]).flatten()

    vv_y = np.array([[i + (m + p) * j for i in range(p)] for j in range(p)]).flatten()
    vv_x = np.array([[i + (n + q) * j for i in range(q)] for j in range(q)]).flatten()

    A_22 = outer_contraction[np.ix_(cv_x + n, cv_y + m)]
    A_33 = outer_contraction[np.ix_(n * (n + q) + vc_x, m * (m + p) + vc_y)]
    A_44 = outer_contraction[np.ix_(vv_x + n + n * (n + q), vv_y + m + m * (m + p))]
    vec_00 = _tt_burer_monteiro_grad(A_22, A_33, A_44, C_00, C_01, C_10, C_11, V_00, V_01, V_10, V_11)
    vec_01 = _tt_burer_monteiro_grad(A_22, A_33, A_44, C_11, C_10, C_01, C_00, V_01, V_00, V_11, V_10)
    vec_10 = _tt_burer_monteiro_grad(A_22, A_33, A_44, C_00, C_01, C_10, C_11, V_10, V_11, V_00, V_01)
    vec_11 = _tt_burer_monteiro_grad(A_22, A_33, A_44, C_11, C_10, C_01, C_00, V_11, V_10, V_01, V_00)

    if 0 < idx < len(cores) - 1:
        cores[idx][:, 0, 0, :] -= lr * vec_00.reshape(xr_i, xr_ip1)
        cores[idx][:, 0, 1, :] -= lr * vec_01.reshape(xr_i, xr_ip1)
        cores[idx][:, 1, 0, :] -= lr * vec_10.reshape(xr_i, xr_ip1)
        cores[idx][:, 1, 1, :] -= lr * vec_11.reshape(xr_i, xr_ip1)
    else:
        r = xr_ip1 if idx == 0 else xr_i
        a = 0 if idx == 0 else -1
        cores[idx][:, 0, 0, :] -= lr * np.expand_dims(np.diagonal(vec_00.reshape(r, r)), axis=a)
        cores[idx][:, 0, 1, :] -= lr * np.expand_dims(np.diagonal(vec_01.reshape(r, r)), axis=a)
        cores[idx][:, 1, 0, :] -= lr * np.expand_dims(np.diagonal(vec_10.reshape(r, r)), axis=a)
        cores[idx][:, 1, 1, :] -= lr * np.expand_dims(np.diagonal(vec_11.reshape(r, r)), axis=a)

    return cores, lr, error


def tt_to_ring(tt_train):
    first_core_top_row = np.hstack((np.diag(tt_train[0][:, 0, 0].flatten()), np.diag(tt_train[0][:, 0, 1].flatten())))
    first_core_bottom_row = np.hstack(
        (np.diag(tt_train[0][:, 1, 0].flatten()), np.diag(tt_train[0][:, 1, 1].flatten())))
    first_core = np.vstack((first_core_top_row, first_core_bottom_row))
    tt_train[0] = first_core

    last_core_top_row = np.hstack((np.diag(tt_train[-1][:, 0, 0].flatten()), np.diag(tt_train[-1][:, 0, 1].flatten())))
    last_core_bottom_row = np.hstack(
        (np.diag(tt_train[-1][:, 1, 0].flatten()), np.diag(tt_train[-1][:, 1, 1].flatten())))
    last_core = np.vstack((last_core_top_row, last_core_bottom_row))
    tt_train[-1] = last_core

    return tt_train


def tt_burer_monteiro_factorisation(psd_tt, max_iter=10):
    tt_train = tt_scale(-1, psd_tt)
    cores = tt_random_gaussian([int(np.ceil(np.sqrt(r))) for r in tt_ranks(tt_train)], shape=(2, 2))
    indices = list(range(len(cores))) + list(reversed(range(len(cores))))
    lr = 0.5 * np.ones(len(cores))
    err = np.inf
    for iteration in range(max_iter):
        for k in indices:
            print(f"Core number {k}")
            cores = tt_rl_orthogonalise_idx(cores, k)
            cores, l, err = _tt_bm_core_wise(tt_train, cores, k, err, lr=lr[k])
            lr[k] = l
    return cores


def tt_infeasible_newton_system(obj_tt, linear_op_tt, bias_tt, X_tt, Y_tt, Z_tt, mu):
    upper_rhs = tt_sub(tt_sub(_tt_linear_op(tt_adjoint(linear_op_tt), Y_tt), Z_tt), obj_tt)
    middle_rhs = tt_sub(_tt_linear_op(linear_op_tt, X_tt), bias_tt)
    lower_rhs = tt_sub(tt_scale(mu, tt_identity(len(X_tt))), tt_linear_op_compose(Z_tt, X_tt))


def _core_op_from_matrix(core):
    mask_1 = np.zeros_like(core)
    mask_2 = np.zeros_like(core)
    mask_3 = np.zeros_like(core)
    mask_4 = np.zeros_like(core)
    mask_1[:, 0, 0] = core[:, 0, 0]
    mask_1[:, 0, 1] = core[:, 1, 0]
    mask_2[:, 0, 0] = core[:, 0, 1]
    mask_2[:, 0, 1] = core[:, 1, 1]
    mask_3[:, 1, 0] = core[:, 0, 0]
    mask_3[:, 1, 1] = core[:, 1, 0]
    mask_4[:, 1, 0] = core[:, 0, 1]
    mask_4[:, 1, 1] = core[:, 1, 1]
    mask_1 = np.expand_dims(mask_1, axis=1)
    mask_2 = np.expand_dims(mask_2, axis=1)
    mask_3 = np.expand_dims(mask_3, axis=1)
    mask_4 = np.expand_dims(mask_4, axis=1)
    res_core = np.concatenate((
        mask_1, mask_2,
        mask_3, mask_4
    ), axis=1)
    return res_core


def tt_op_from_tt_matrix(tt_train):
    return [_core_op_from_matrix(c) for c in tt_train]


def tt_adjoint(linear_op_tt):
    return [np.swapaxes(c, axis1=2, axis2=3) for c in linear_op_tt]
