import numpy as np
from typing import List, Tuple
from itertools import product
from copy import deepcopy
from tqdm import tqdm

PHI = np.array([[1, 1],
                [1, -1]], dtype=float).reshape(1, 2, 2, 1)

PHI_INV = np.array([[1 / 2, 1 / 2],
                    [1 / 2, -1 / 2]], dtype=float).reshape(1, 2, 2, 1)

I = np.array([[1, 0],
              [0, 1]]).reshape(1, 2, 2, 1)


def tt_indentity(dim):
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


def safe_multi_dot(matrices: List[np.array]):
    if len(matrices) > 1:
        return np.linalg.multi_dot(matrices)
    elif len(matrices) == 1:
        return matrices[0]
    return []


def tt_random_binary(target_ranks: List[int]):
    target_ranks = [1] + target_ranks + [1]
    return tt_normalise([2 * (np.random.randint(low=0, high=2, size=(l_n, 2, l_np1)) - 0.5) for l_n, l_np1 in
                         zip(target_ranks[:-1], target_ranks[1:])])


def tt_random_gaussian_matrix(target_ranks: List[int]):
    target_ranks = [1] + target_ranks + [1]
    return tt_normalise([1 / (l_n * 2 * 2 * l_np1) * np.random.randn(l_n, 2, 2, l_np1) for l_n, l_np1 in
                         zip(target_ranks[:-1], target_ranks[1:])])


def tt_one(dim):
    """ Returns an all-one tensor of dimension 2**dim """
    return [np.ones((1, 2, 1)) for _ in range(dim)]


def tt_one_bonded(dim, bond_dim):
    """ Returns an all-one tensor of dimension 2**dim with one bonded dimension"""
    one = [np.ones((1, 2, 1)) for _ in range(dim - 2)]
    one.insert(bond_dim, np.ones((1, 2, 2, 1)))
    return one


def tt_leading_one(dim):
    """ Returns a tensor of dimension 2**dim with a one entry in its 00000...00-entry"""
    return tt_atom_train(dim + 1, dim)


def tt_atom_train(idx, dim):
    """ Returns a tensor of dimension 2**dim with a one entry at the respective degree 1 monomial"""
    return [
        np.array([0, 1], dtype=float).reshape(1, 2, 1) if idx == i else np.array([1, 0], dtype=float).reshape(1, 2, 1)
        for i in range(dim)
    ]


def tt_rl_orthogonalize(tt_train: List[np.array]):
    for idx in range(len(tt_train) - 1, 0, -1):
        shape_p1 = tt_train[idx].shape
        shape = tt_train[idx - 1].shape
        Q_T, R = np.linalg.qr(tt_train[idx].reshape(shape_p1[0], -1).T)
        tt_train[idx] = Q_T.T.reshape(-1, *shape_p1[1:-1], shape_p1[-1])
        tt_train[idx - 1] = (tt_train[idx - 1].reshape(-1, R.shape[-1]) @ R.T).reshape(-1, *shape[1:-1],
                                                                                       tt_train[idx].shape[0])
    return tt_train


def tt_lr_orthogonalize(tt_train: List[np.array]):
    tt_train = tt_swap_all(tt_train)
    tt_train = tt_rl_orthogonalize(tt_train)
    tt_train = tt_swap_all(tt_train)
    return tt_train


def tt_bond_at(tt_train, idx):
    if idx != -1:
        tt_train = tt_train[:idx] + [np.einsum("abc, cde -> abde", tt_train[idx], tt_train[idx + 1])] + tt_train[
                                                                                                        idx + 2:]
    return tt_train


def core_bond(core_1, core_2):
    return np.einsum("abc, cde -> abde", core_1, core_2)


def break_core_bond(core):
    """ Breaks up a bond between two cores """
    shape = core.shape
    A = core.reshape(shape[0] * shape[1], -1)
    U, S, V_T = np.linalg.svd(A)
    non_sing_eig_idxs = np.asarray(np.abs(S) > 0).nonzero()
    S = S[non_sing_eig_idxs]
    next_rank = len(S)
    U = U[:, non_sing_eig_idxs]
    V_T = V_T[non_sing_eig_idxs, :]
    G_i = U.reshape(shape[0], shape[1], next_rank)
    G_ip1 = (np.diag(S) @ V_T).reshape(next_rank, shape[2], shape[-1])
    return [G_i, G_ip1]


def tt_inf_schatten_norm(tt_train):
    """Loss criterion on the TT-rank"""
    tt_train = tt_train + [np.array([1, 0]).reshape(1, 2, 1)] if len(tt_train) % 2 == 1 else tt_train
    tt_train = tt_walsh_op(tt_train)
    return np.product(np.array([  # TODO: prod or max or logsum?
        np.linalg.norm(np.einsum("abc, cde -> abde", tt_train[idx], tt_train[idx + 1]).reshape(
            tt_train[idx].shape[0] * tt_train[idx + 1].shape[-1], -1), ord=2) for idx in range(0, len(tt_train) - 1, 2)
        # largest singular value
    ]))


def tt_nuc_schatten_norm(tt_train):
    """Loss criterion on the TT-rank"""
    tt_train = tt_train + [np.array([1, 0]).reshape(1, 2, 1)] if len(tt_train) % 2 == 1 else tt_train
    tt_train = tt_walsh_op(tt_train)
    return np.max(np.array([  # TODO: prod or max or logsum?
        np.linalg.norm(np.einsum("abc, cde -> abde", tt_train[idx], tt_train[idx + 1]).reshape(
            tt_train[idx].shape[0] * tt_train[idx + 1].shape[-1], -1), ord='nuc') for idx in
        range(0, len(tt_train) - 1, 2)
        # largest singular value
    ]))


def tt_rl_contraction(tt_train_1: List[np.array], tt_train_2: List[np.array]):
    # TODO: Adapt for linear_ops
    new_cores = [
        tt_train_1[-1].reshape(tt_train_1[-1].shape[0], -1) @ tt_train_2[-1].reshape(tt_train_2[-1].shape[0], -1).T]
    for core_1, core_2 in zip(tt_train_1[-2:0:-1], tt_train_2[-2:0:-1]):
        core_w = new_cores[-1]
        core_z = core_1.reshape(-1, core_w.shape[0]) @ core_w
        new_cores.append(core_z.reshape(core_1.shape[0], -1) @ core_2.reshape(core_2.shape[0], -1).T)
    return new_cores[::-1]


def tt_randomise_orthogonalise(tt_train: List[np.array], target_ranks: List[int]) -> List[np.array]:
    # TODO: Adapt for linear_ops
    if len(tt_train) > 1:
        target_ranks = [1] + target_ranks + [1]
        tt_gaussian = [
            (1 / (l_n * np.prod(tt_train[i].shape[1:-1]) * l_np1)) * np.random.randn(l_n, *tt_train[i].shape[1:-1],
                                                                                     l_np1)
            for i, (l_n, l_np1) in enumerate(zip(target_ranks[:-1], target_ranks[1:]))
        ]
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


def tt_rank_reduce(tt_train: List[np.array], tt_bound=None):
    """ Might reduce TT-rank """
    # TODO: Adapt for linear_ops
    tt_train = tt_rl_orthogonalize(tt_train)
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


def tt_rank(tt_train):
    return np.max([max(t.shape[0], t.shape[-1]) for t in tt_train])


def tt_ranks(tt_train):
    return [t.shape[-1] for t in tt_train[:-1]]


def tt_rank_retraction(tt_upper_ranks: List[int], tt_train: List[np.array]):
    """ Might reduce TT-rank """
    tt_train = tt_rl_orthogonalize(tt_train)
    rank = 1
    for idx, upper_rank in enumerate(tt_upper_ranks):
        idx_shape = tt_train[idx].shape
        next_idx_shape = tt_train[idx + 1].shape
        U, S, V_T = np.linalg.svd(tt_train[idx].reshape(rank * idx_shape[1], -1))
        abs_S = np.abs(S)
        next_rank = min(upper_rank, len(abs_S > 0))
        non_sing_eig_idxs = np.argpartition(abs_S, -next_rank)[-next_rank:]
        S = S[non_sing_eig_idxs]
        U = U[:, non_sing_eig_idxs]
        V_T = V_T[non_sing_eig_idxs, :]
        tt_train[idx] = U.reshape(rank, idx_shape[1], next_rank)
        tt_train[idx + 1] = (np.diag(S) @ V_T @ tt_train[idx + 1].reshape(V_T.shape[-1], -1)).reshape(next_rank,
                                                                                                      next_idx_shape[1],
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


def tt_kronecker_prod(tt_train_1: List[np.array], tt_train_2: List[np.array]) -> List[np.array]:
    return tt_train_1 + tt_train_2


def bool_to_tt_train(bool_values: List[bool]):
    """
    Converts a list of boolean values into its respective tensor train
    """
    return tt_walsh_op_inv([np.array([1, 2 * float(b_value) - 1]).reshape(1, 2, 1) for b_value in bool_values])


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


def tt_grad_inner_prod(tt_train_1: List[np.array], tt_train_2: List[np.array], gradient_core: np.array, idx):
    return np.sum(
        np.linalg.multi_dot(
            [_tt_core_collapse(core_1, core_2) for core_1, core_2 in zip(tt_train_1[:idx], tt_train_2[:idx])]
            + [_tt_core_collapse(tt_train_1[idx], gradient_core)]
            + [_tt_core_collapse(core_1, core_2) for core_1, core_2 in zip(tt_train_1[idx + 1:], tt_train_2[idx + 1:])]
        )
    )


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
    Produces the truth table result tensor
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
    Produces the truth table result tensor
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


def tt_boolean_criterion(tt_train: List[np.array]) -> float:
    tt_train = tt_walsh_op(tt_train)
    tt_train_xnor = tt_hadamard(tt_train, tt_train)
    tt_train_xnor = tt_walsh_op_inv(tt_train_xnor)
    return np.abs(tt_inner_prod(tt_train_xnor, tt_train_xnor) - 1)


def tt_extract_seq(tt_train, assignments):
    N = len(tt_train)
    eps = 0.5 ** (1 / N) + 1 / 2 ** N
    tt_ttable = tt_rl_orthogonalize(tt_walsh_op(tt_train))
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
    tt_train_1 = tt_walsh_op(tt_train_1)
    tt_train_2 = tt_walsh_op(tt_train_2)
    tt_train_xnor = tt_hadamard(tt_train_1, tt_train_2)
    tt_train_xnor = tt_walsh_op_inv(tt_train_xnor)
    return tt_rank_reduce(tt_train_xnor)


def tt_xor(tt_train_1: List[np.array], tt_train_2: List[np.array]) -> List[np.array]:
    return tt_xnor(tt_neg(tt_train_1), tt_train_2)


def tt_and(tt_train_1: List[np.array], tt_train_2: List[np.array]) -> List[np.array]:
    tt_train_1 = tt_walsh_op(tt_train_1)
    tt_train_2 = tt_walsh_op(tt_train_2)
    tt_train_1 = tt_scale(0.5, tt_train_1)
    tt_mul = tt_hadamard(tt_train_1, tt_train_2)
    tt_train_2 = tt_scale(0.5, tt_train_2)
    half = tt_scale(-0.5, tt_one(len(tt_train_1)))
    tt_train_and = tt_add(tt_add(half, tt_mul), tt_add(tt_train_1, tt_train_2))
    tt_train_and = tt_walsh_op_inv(tt_train_and)
    return tt_rank_reduce(tt_train_and)


def tt_or(tt_train_1: List[np.array], tt_train_2: List[np.array]) -> List[np.array]:
    return tt_neg(tt_and(tt_neg(tt_train_1), tt_neg(tt_train_2)))


def tt_neg(tt_train: List[np.array]) -> List[np.array]:
    return tt_scale(-1, tt_train)


def tt_scale(alpha, tt_train):
    idx = np.random.randint(low=0, high=len(tt_train))
    mul = alpha * tt_train[idx]
    return tt_train[:idx] + [mul] + tt_train[idx + 1:]


def tt_normalise(tt_train, radius=1):
    return tt_scale(np.divide(radius, np.sqrt(tt_inner_prod(tt_train, tt_train))), tt_train)


def tt_add_noise(tt_train, target_ranks):
    target_ranks = [1] + target_ranks + [1]
    # approximately uniform noise on the sphere
    noise_train = [(1 / (l_n * 2 * l_np1)) * np.random.randn(l_n, 2, l_np1) for l_n, l_np1 in
                   zip(target_ranks[:-1], target_ranks[1:])]
    # projection onto tangent space of tt_train
    tt_train = tt_scale(1 - tt_inner_prod(noise_train, tt_train), tt_train)
    tt_train = tt_add(tt_train, noise_train)
    return tt_normalise(tt_train)


def tt_round_iteration(tt_train):
    tt_train_p3 = tt_scale(-0.5, tt_hadamard(tt_hadamard(tt_train, tt_train), tt_train))
    tt_train = tt_scale(1.5, tt_train)
    tt_train = tt_add(tt_train, tt_train_p3)
    return tt_rank_reduce(tt_train)


def tt_round(tt_train, iterations=100):
    for _ in range(iterations):
        tt_train = tt_round_iteration(tt_train)
    return tt_round_iteration(tt_train)


def tt_abs(tt_train):
    rounded_tt_train = tt_round(tt_train)
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


def tt_graph_to_tensor(n, edges):  # Start numbering nodes at 0
    tensor = -np.ones([2] * n + [2] * n)
    for e in edges:
        index_1 = [int(x) for x in reversed(bin(e[0])[2:])]
        index_1 += [0] * (n - len(index_1))
        index_2 = [int(x) for x in reversed(bin(e[1])[2:])]
        index_2 += [0] * (n - len(index_2))
        tensor[tuple(index_1 + index_2)] = 1
        tensor[tuple(index_2 + index_1)] = 1
    return tensor


def tt_tensor_matrix(tt_trains: List[np.array]):
    n = len(tt_trains)
    bin_length = len(bin(n - 1)[2:])
    binary_i = '0' * bin_length
    index = [(np.expand_dims(np.zeros_like(c), 1), np.expand_dims(c, 1)) if bool(int(b)) else (
        np.expand_dims(c, 1), np.expand_dims(np.zeros_like(c), 1)) for b, c in
             zip(binary_i, tt_trains[0][-bin_length:])]
    tt_tensor_matrix = tt_trains[0][:-bin_length] + [np.concatenate(tup, axis=1) for tup in index]
    for i in range(1, n):
        binary_i = bin(i)[2:]
        binary_i = '0' * (bin_length - len(binary_i)) + binary_i
        index = [(np.expand_dims(np.zeros_like(c), 1), np.expand_dims(c, 1)) if bool(int(b)) else (
            np.expand_dims(c, 1), np.expand_dims(np.zeros_like(c), 1)) for
                 b, c in zip(binary_i, tt_trains[i][-bin_length:])]
        current_column = tt_trains[i][:-bin_length] + [np.concatenate(tup, axis=1) for tup in index]
        tt_tensor_matrix = tt_add(tt_tensor_matrix, current_column)
        tt_tensor_matrix = tt_rank_reduce(tt_tensor_matrix)

    return tt_tensor_matrix, bin_length


def tt_conjugate_gradient(linear_op: List[np.array], tt_train: List[np.array], num_iter=10, tol=1e-7):
    x = [np.zeros((1, 2, 1)) for _ in range(len(tt_train))]
    r = p = tt_train
    r_2 = tt_inner_prod(r, r)
    for _ in range(num_iter):
        prev_r_2 = r_2
        l_op_p = tt_linear_op(linear_op, p)
        p_rest = tt_inner_prod(p, l_op_p)
        alpha = np.divide(prev_r_2, p_rest)
        r = tt_rl_orthogonalize(tt_add(r, tt_scale(-alpha, l_op_p, idx=np.random.randint(low=0, high=len(p)))))
        r_2 = tt_inner_prod(r, r)
        beta = np.divide(r_2, prev_r_2)
        x = tt_rl_orthogonalize(tt_add(x, tt_scale(alpha, p, idx=np.random.randint(low=0, high=len(p)))))
        if np.less_equal(r_2, tol):
            break
        p = tt_rl_orthogonalize(tt_add(r, tt_scale(beta, p, idx=np.random.randint(low=0, high=len(p)))))
    return tt_rank_reduce(x)


def tt_swap_all(tt_train: List[np.array]):
    return [np.swapaxes(c, 0, -1) for c in reversed(tt_train)]


def tt_max_eigentensor(linear_op: List[np.array], num_iter=10):
    normalisation = np.sqrt(tt_inner_prod(linear_op, linear_op))
    linear_op = tt_scale(-(1 / normalisation), linear_op)
    eig_vec = tt_normalise([np.random.randn(1, 2, 1) for _ in range(len(linear_op))])
    for _ in range(num_iter):
        eig_vec = tt_linear_op(linear_op, eig_vec)
        eig_vec = tt_normalise(eig_vec)
    prev_eig_vec = eig_vec
    eig_vec = tt_linear_op(linear_op, eig_vec)
    eig_val = tt_inner_prod(prev_eig_vec, eig_vec)
    return tt_normalise(eig_vec), normalisation*eig_val


def _tt_op_core_collapse(linear_core_op: np.array, core_2: np.array) -> np.array:
    return sum([
        np.kron(linear_core_op[:, :, i], core_2[:, None, i])
        for i in [0, 1]
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


def tt_linear_op(linear_op, tt_train: List[np.array]) -> List[np.array]:
    return tt_rl_orthogonalize(_tt_linear_op(linear_op, tt_train))


def tt_rank_reduced_linear_op(linear_op, tt_train: List[np.array], bound=None) -> List[np.array]:
    tt_train = _tt_linear_op(linear_op, tt_train)
    return tt_rank_reduce(tt_train, bound)


def tt_randomised_linear_op(linear_op, tt_train: List[np.array], ranks) -> List[np.array]:
    return tt_randomise_orthogonalise(_tt_linear_op(linear_op, tt_train), target_ranks=ranks)


def _tt_op_op_collapse(linear_core_op_1, linear_core_op_2):
    return np.concatenate([
        np.expand_dims(_tt_op_core_collapse(linear_core_op_1, linear_core_op_2[:, :, i]), 1)
        for i in [0, 1]
    ], axis=1)


def tt_linear_op_compose(tt_linear_op_1, tt_linear_op_2):
    split_idx = np.argmax([len(c.shape) for c in tt_linear_op_1])
    half_core = safe_multi_dot([
        _tt_core_collapse(core_op_1, core_op_2) for core_op_1, core_op_2 in
        zip(tt_linear_op_1[:split_idx], tt_linear_op_2[:split_idx])
    ])
    full_cores = [
        _tt_op_op_collapse(core_op_1, core_op_2) for core_op_1, core_op_2 in
        zip(tt_linear_op_1[split_idx:], tt_linear_op_2[split_idx:])
    ]
    if len(half_core) > 0:
        full_cores[0] = np.einsum("ab, bcde -> acde", half_core, full_cores[0])
    return tt_transpose(full_cores)


def tt_transpose(tt_linear_op):
    split_idx = np.argmax([len(c.shape) for c in tt_linear_op])
    return tt_linear_op[:split_idx] + [np.swapaxes(c, axis1=1, axis2=2) for c in tt_linear_op[split_idx:]]


def tt_gram(tt_linear_op):
    tt_linear_op_t = tt_transpose(tt_linear_op)
    gram = tt_linear_op_compose(tt_linear_op, tt_linear_op_t)
    gram = tt_rank_reduce(gram)
    return gram


def tt_trace(linear_op):
    I = tt_indentity(len(linear_op))
    return tt_inner_prod(linear_op, I)


def tt_min_eigentensor(linear_op: List[np.array], num_iter=10, tol=1e-3):
    n = len(linear_op)
    normalisation = np.sqrt(tt_inner_prod(linear_op, linear_op))
    linear_op = tt_scale(-(1 / normalisation), linear_op)
    identity = [I for _ in range(n)]
    identity = tt_scale(2, identity)
    linear_op = tt_add(identity, linear_op)
    linear_op = tt_rank_reduce(linear_op, tt_bound=0)
    eig_vec = tt_normalise([np.random.randn(1, 2, 1) for _ in range(n)])
    norm_2 = np.inf
    for i in range(num_iter):
        prev_norm_2 = norm_2
        eig_vec = tt_rank_reduced_linear_op(linear_op, eig_vec, bound=1e-12)
        norm_2 = tt_inner_prod(eig_vec, eig_vec)
        eig_vec = tt_scale(np.divide(1, np.sqrt(norm_2)), eig_vec)
        if np.less_equal(np.abs(norm_2 - prev_norm_2), tol):
            break
    prev_eig_vec = eig_vec
    eig_vec = tt_rank_reduced_linear_op(linear_op, eig_vec, bound=1e-12)
    eig_val = tt_inner_prod(prev_eig_vec, eig_vec)
    return tt_normalise(eig_vec), normalisation * (2 - eig_val)


def tt_randomised_min_eigentensor(linear_op: List[np.array], num_iter=10, tol=1e-3):
    n = len(linear_op)
    normalisation = np.sqrt(tt_inner_prod(linear_op, linear_op))
    linear_op = tt_scale(-np.divide(1, normalisation), linear_op)
    identity = [I for _ in range(n)]
    identity = tt_scale(2, identity)
    linear_op = tt_rank_reduce(tt_add(identity, linear_op), tt_bound=0)
    eig_vec = tt_normalise([np.random.randn(1, 2, 1) for _ in range(n)])
    ranks = tt_ranks(linear_op)
    norm_2 = np.inf
    for i in range(num_iter):
        prev_norm_2 = norm_2
        eig_vec = tt_randomised_linear_op(linear_op, eig_vec, ranks)
        norm_2 = tt_inner_prod(eig_vec, eig_vec)
        eig_vec = tt_scale(np.divide(1, np.sqrt(norm_2)), eig_vec)
        if np.less_equal(np.abs(norm_2 - prev_norm_2), tol):
            break
    prev_eig_vec = eig_vec
    eig_vec = tt_rank_reduced_linear_op(linear_op, eig_vec)
    eig_val = tt_inner_prod(prev_eig_vec, eig_vec)
    return tt_normalise(eig_vec), normalisation * (2 - eig_val)


def _trivial_step_size(it):
    return np.minimum(1.0, 2.0 / it), it + 1


def _bt_step_size(sdp_gradient, eig_sdp, lag_mul_1, lag_mul_2, obj_sdp, linear_op_sdp, bias, X, M, eta=1.1,
                  max_iter=10):
    ATy = tt_constraint_contract(linear_op_sdp, lag_mul_1)
    g = -tt_inner_prod(tt_normalise(sdp_gradient), eig_sdp)
    Ax = tt_inner_prod(ATy, X)
    obj_1 = tt_inner_prod(obj_sdp, X)
    obj_2 = lag_mul_1 * (Ax - bias)
    Ad = tt_inner_prod(ATy, eig_sdp)
    update_1 = tt_inner_prod(obj_sdp, eig_sdp)
    update_2 = lag_mul_1 * (Ad - bias)

    Lag_new = lambda gamma: (1 - gamma) * (obj_1 + obj_2) + gamma * (update_1 + update_2) + (lag_mul_2 / 2) * (
        (1 - gamma) * Ax + gamma * Ad - bias) ** 2
    gamma = np.minimum(1.0, g / M)
    current_obj = Lag_new(gamma)
    for i in range(max_iter):
        M *= eta
        gamma = np.minimum(1.0, g / M)
        prev_obj = current_obj
        current_obj = Lag_new(gamma)
        if np.less_equal(prev_obj, current_obj) or gamma == 1:
            M *= 1 / eta
            gamma = np.minimum(1.0, g / M)
            break
    return gamma, M + 1


def _tt_lag_collapse(linear_core_op: np.array, core_2: np.array) -> np.array:
    return sum([
        np.kron(linear_core_op[:, i], core_2[:, i, None, None])
        for i in [0, 1]
    ])


def tt_constraint_contract(linear_op, lag_mul):
    split_idx = np.argmax([len(c.shape) for c in linear_op])
    return linear_op[:split_idx] + [_tt_lag_collapse(op, l) for op, l in zip(linear_op[split_idx:], lag_mul)]


def tt_outer_product(tt_train_1, tt_train_2):
    return tt_rank_reduce(
        [np.kron(np.expand_dims(c_1, 1), np.expand_dims(c_2, 2)) for c_1, c_2 in zip(tt_train_1, tt_train_2)])


def tt_binary_round(tt_train, tol=1e-3):
    for _ in range(100):
        tt_table = tt_walsh_op(tt_train)
        tt_table_p2 = tt_rank_reduce(tt_hadamard(tt_table, tt_table))
        boolean_poly = tt_walsh_op_inv(tt_table_p2)
        if np.less_equal(np.abs(tt_inner_prod(boolean_poly, boolean_poly) - 1), tol):
            break
        tt_table_p3 = tt_rank_reduce(tt_hadamard(tt_table_p2, tt_table))
        tt_update = tt_scale(-0.5, tt_walsh_op_inv(tt_table_p3))
        next_tt_train = tt_scale(1 - tt_inner_prod(tt_update, tt_train), tt_train)
        next_tt_train = tt_rank_reduce(tt_add(next_tt_train, tt_update))
        tt_train = tt_normalise(next_tt_train)
    return tt_train


def tt_random_graph(target_ranks):
    matrix = tt_random_gaussian_matrix([int(np.ceil(r / 2)) for r in target_ranks])
    matrix_t = tt_transpose(matrix)
    symmetric_matrix = tt_rank_reduce(tt_add(matrix, matrix_t))
    I_tensor = sum([break_core_bond(I) for _ in symmetric_matrix], [])
    mask = tt_add(tt_one(len(I_tensor)), tt_scale(-1, I_tensor))
    symmetric_matrix = sum([break_core_bond(core) for core in symmetric_matrix], [])
    symmetric_matrix = tt_normalise(symmetric_matrix)
    symmetric_matrix = tt_binary_round(symmetric_matrix)
    symmetric_matrix = tt_walsh_op(symmetric_matrix)
    symmetric_matrix = tt_hadamard(symmetric_matrix, mask)
    symmetric_matrix = tt_add(symmetric_matrix, tt_scale(-1, I_tensor))
    return tt_rank_reduce([core_bond(c_1, c_2) for c_1, c_2 in zip(symmetric_matrix[:-1:2], symmetric_matrix[1::2])])


def tt_op_to_matrix(linear_op):
    tensor = tt_to_tensor(linear_op)
    n = 2**(len(tensor.shape)//2)
    bin_length = len(bin(n - 1)[2:])
    binary_i = '0' * bin_length
    matrix = tensor[tuple(sum([[int(i), slice(None)] for i in binary_i], []))].reshape(-1, 1)
    for i in range(1, n):
        binary_i = bin(i)[2:]
        binary_i = '0' * (bin_length - len(binary_i)) + binary_i
        index = tuple(sum([[int(i), slice(None)] for i in binary_i], []))
        matrix = np.hstack((matrix, tensor[index].reshape(-1, 1)))
    return matrix


def _cg_oracle(lag_mul_1, lag_mul_2, obj_sdp, linear_op_sdp, res):
    ATy = tt_constraint_contract(linear_op_sdp, lag_mul_1)
    Ares = tt_scale(lag_mul_2, tt_constraint_contract(linear_op_sdp, res))
    constraint_term = tt_rank_reduce(tt_add(ATy, Ares))
    sdp_gradient = tt_rank_reduce(tt_add(obj_sdp, constraint_term))
    tt_eig, _ = tt_randomised_min_eigentensor(sdp_gradient, num_iter=100)
    eig_sdp = tt_outer_product(tt_eig, tt_eig)
    eig_sdp = tt_normalise(eig_sdp)
    return eig_sdp, sdp_gradient


def _interpolate(gamma, X, eig_sdp, trace_param):
    tt_update = tt_scale(gamma*trace_param, eig_sdp)
    X = tt_scale(1 - gamma, X)
    X = tt_add(X, tt_update)
    return tt_rank_reduce(X)


def _tt_cg(lag_mul_1, lag_mul_2, obj_sdp, linear_op_sdp, _, res, X, M, trace_param):
    # TODO: lagrangian_mul will become tensor
    # TODO: Need to move reg term with lag_mul_2 when we have multiple constraints
    eig_sdp, _ = _cg_oracle(lag_mul_1, lag_mul_2, obj_sdp, linear_op_sdp, res)
    gamma, M = _trivial_step_size(M)  # take square root as update is tt_eig tt_eig.T
    X = _interpolate(gamma, X, eig_sdp, trace_param)
    return X, M


def _tt_cg_bt(lag_mul_1, lag_mul_2, obj_sdp, linear_op_sdp, bias, res, X, M, trace_param):
    eig_sdp, sdp_gradient = _cg_oracle(lag_mul_1, lag_mul_2, obj_sdp, linear_op_sdp, res)
    gamma, M = _bt_step_size(sdp_gradient, eig_sdp, lag_mul_1, lag_mul_2, obj_sdp, linear_op_sdp, bias, X, M)
    X = _interpolate(gamma, X, eig_sdp, trace_param)
    return X, M


def _tt_lag_core_collapse(linear_core_op: np.array, core_2: np.array) -> np.array:
    return sum([
        np.kron(linear_core_op[:, :, i, j], core_2[:, None, i, j])
        for (i, j) in product([0, 1], [0, 1])
    ])


def tt_eval_constraints(linear_op, X):
    split_idx = np.argmax([len(c.shape) for c in linear_op])
    factor = safe_multi_dot(
        [_tt_core_collapse(core_1, core_2) for core_1, core_2 in zip(linear_op[:split_idx], X[:split_idx])])
    full_cores = [_tt_lag_core_collapse(core_1, core_2) for core_1, core_2 in zip(linear_op[split_idx:], X[split_idx:])]
    if len(factor) > 0:
        full_cores[0] = np.einsum("ab, bcd -> acd", factor, full_cores[0])
    return full_cores


def tt_sdp_fw(obj_sdp: List[np.array], linear_ops: List[np.array], bias: List[np.array], trace_param=1, bt=True, num_iter=10):
    M = 1
    cg_func = _tt_cg_bt if bt else _tt_cg
    X = [np.zeros((1, 2, 2, 1)) for _ in range(len(linear_ops))]
    neg_bias = tt_scale(-1, bias)
    res = neg_bias
    lag_mul_1 = [np.zeros((1, 2, 1)) for _ in range(len(bias))]
    for it in range(1, num_iter + 1):
        lag_mul_2 = np.sqrt(M + 1)
        X, M = cg_func(lag_mul_1, lag_mul_2, obj_sdp, linear_ops, bias, res, X, M, trace_param)
        res = tt_add(tt_eval_constraints(linear_ops, X), neg_bias)
        alpha = np.divide(2, np.power(M, 1/2))
        lag_mul_1 = tt_rank_reduce(tt_add(lag_mul_1, tt_scale(alpha, res)), tt_bound=0)
    return X


