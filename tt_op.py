from copy import deepcopy

import jax
import numpy as np
import jax.numpy as jnp
from typing import List
from itertools import product

PHI = np.array([[1, 1],
                [1, -1]], dtype=float).reshape(1, 2, 2, 1)

PHI_INV = np.array([[1 / 2, 1 / 2],
                    [1 / 2, -1 / 2]], dtype=float).reshape(1, 2, 2, 1)


def phi(num_bonds):
    """ Bonds "num_bonds" PHI cores together """
    tensor = PHI
    new_PHI = PHI
    for _ in range(2 ** (num_bonds) - 1):
        new_PHI = jnp.expand_dims(new_PHI, (0, 1))
        tensor = jnp.kron(jnp.expand_dims(tensor, (-1, -2)), new_PHI)
    return tensor


def phi_inv(num_bonds):
    """ Bonds "num_bonds" PHI cores together """
    tensor = PHI_INV
    new_PHI = PHI_INV
    for _ in range(2 ** (num_bonds) - 1):
        new_PHI = jnp.expand_dims(new_PHI, (0, 1))
        tensor = jnp.kron(jnp.expand_dims(tensor, (-1, -2)), new_PHI)
    return tensor


def tt_one(dim):
    """ Returns an all-one tensor of dimension 2**dim """
    return [jnp.ones((1, 2, 1)) for _ in range(dim)]


def tt_one_bonded(dim, bond_dim):
    """ Returns an all-one tensor of dimension 2**dim with one bonded dimension"""
    one = [jnp.ones((1, 2, 1)) for _ in range(dim - 2)]
    one.insert(bond_dim, jnp.ones((1, 2, 2, 1)))
    return one


def tt_leading_one(dim):
    """ Returns a tensor of dimension 2**dim with a one entry in its 00000...00-entry"""
    return tt_atom_train(dim + 1, dim)


def tt_atom_train(idx, dim):
    """ Returns a tensor of dimension 2**dim with a one entry at the respective degree 1 monomial"""
    return [
        jnp.array([0, 1], dtype=float).reshape(1, 2, 1) if idx == i else jnp.array([1, 0], dtype=float).reshape(1, 2, 1)
        for i in range(dim)
    ]


def tt_rl_orthogonalize(tt_train: List[np.array]):
    for idx in range(len(tt_train) - 1, 0, -1):
        shape_p1 = tt_train[idx].shape
        shape = tt_train[idx - 1].shape
        Q_T, R = jnp.linalg.qr(tt_train[idx].reshape(shape_p1[0], -1).T)
        tt_train[idx] = Q_T.T.reshape(-1, shape_p1[1], shape_p1[-1])
        tt_train[idx - 1] = (tt_train[idx - 1].reshape(-1, R.shape[-1]) @ R.T).reshape(-1, shape[1],
                                                                                       tt_train[idx].shape[0])
    return tt_train


def part_bond(core):
    """ Breaks up a bond between two cores """
    shape = core.shape
    A = core.reshape(shape[0] * shape[1], -1)
    U, S, V_T = jnp.linalg.svd(A)
    non_sing_eig_idxs = jnp.asarray(np.abs(S) > 0).nonzero()
    S = S[non_sing_eig_idxs]
    next_rank = len(S)
    U = U[:, non_sing_eig_idxs]
    V_T = V_T[non_sing_eig_idxs, :]
    G_i = U.reshape(shape[0], shape[1], next_rank)
    G_ip1 = (jnp.diag(S) @ V_T).reshape(next_rank, shape[2], shape[-1])
    return G_i, G_ip1


def tt_rank_reduce(tt_train: List[np.array]):
    """ Might reduce TT-rank """
    tt_train = tt_rl_orthogonalize(tt_train)
    rank = 1
    for idx in range(len(tt_train) - 1):
        idx_shape = tt_train[idx].shape
        next_idx_shape = tt_train[idx + 1].shape
        U, S, V_T = np.linalg.svd(tt_train[idx].reshape(rank * idx_shape[1], -1))
        non_sing_eig_idxs = np.asarray(np.abs(S) > 1e-5).nonzero()
        S = S[non_sing_eig_idxs]
        next_rank = len(S)
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


def tt_to_tensor(tt_train: List[np.array]) -> np.array:
    """
    Converts a tensor train back into a tensor
    """
    fourier_tensor = np.zeros([t.shape[1] for t in tt_train])
    multi_idxs = product(*[list(range(t.shape[1])) for t in tt_train])
    for idx in multi_idxs:
        dot_prod = tt_train[0][:, idx[0], :]
        for i, core in enumerate(tt_train[1:]):
            dot_prod = dot_prod @ core[:, idx[i + 1], :]
        fourier_tensor[idx] = dot_prod
    return fourier_tensor


def tt_leading_entry(tt_train: List[np.array]) -> np.array:
    """
    Returns the leading entry of a TT-train
    """
    return jnp.sum(
        jnp.linalg.multi_dot([core[tuple([slice(None)] + [0] * (len(core.shape) - 2))] for core in tt_train]))


def _block_diag_tensor(tensor_1: np.array, tensor_2: np.array) -> np.array:
    """
    For internal use: Concatenates two tensors to a block diagonal tensor
    """
    column_1 = jnp.concatenate((tensor_1, np.zeros((tensor_2.shape[0],) + tensor_1.shape[1:])), axis=0)
    column_2 = jnp.concatenate((jnp.zeros((tensor_1.shape[0],) + tensor_2.shape[1:]), tensor_2), axis=0)
    return jnp.concatenate((column_1, column_2), axis=-1)


def tt_add(tt_train_1: List[np.array], tt_train_2: List[np.array]) -> List[np.array]:
    """
    Adds two tensor trains
    """
    return [jnp.concatenate((tt_train_1[0], tt_train_2[0]), axis=-1)] + \
        [_block_diag_tensor(core_1, core_2) for core_1, core_2 in zip(tt_train_1[1:-1], tt_train_2[1:-1])] + \
        [jnp.concatenate((tt_train_1[-1], tt_train_2[-1]), axis=0)]


def _tt_train_kron(core_1: np.array, core_2: np.array) -> np.array:
    """
    For internal use: Computes the kronecker product between two TT-cores with appropriate dimensional
    expansion
    """
    layers = []
    core_shape_length = len(core_1.shape)
    axes = list(range(1, core_shape_length - 1))
    for i in product(*([[0, 1]] * (core_shape_length - 2))):
        idx = (slice(None),) + i
        layers.append(jnp.kron(jnp.expand_dims(core_1[idx], axis=axes), jnp.expand_dims(core_2[idx], axis=axes)))
    return jnp.concatenate(layers, axis=1).reshape(
        (core_1.shape[0] * core_2.shape[0],) + core_1.shape[1:-1] + (core_1.shape[-1] * core_2.shape[-1],)
    )


def tt_hadamard(tt_train_1: List[np.array], tt_train_2: List[np.array]) -> List[np.array]:
    """
    Computes the hadamard product/pointwise multiplication of two tensor trains
    """
    return [_tt_train_kron(core_1, core_2) for core_1, core_2 in zip(tt_train_1, tt_train_2)]


def bool_to_tt_train(bool_values: List[bool]):
    """
    Converts a list of boolean values into its respective tensor train
    """
    return [jnp.array([1, 2 * float(b_value) - 1]).reshape(1, -1, 1) for b_value in bool_values]


def _tt_core_collapse(core_1: np.array, core_2: np.array):
    return sum([
        jnp.kron(core_1[(slice(None),) + i], core_2[(slice(None),) + i])
        for i in product(*([[0, 1]] * (len(core_1.shape) - 2)))
    ])


def bond_at(e, idx):
    if idx != -1:
        e_bond = jnp.einsum("abc, cde -> abde", e[idx], e[idx + 1])
        e = e[:idx] + [e_bond] + e[idx + 2:]
    return e


def tt_inner_prod(tt_train_1: List[np.array], tt_train_2: List[np.array]) -> float:
    """
    Computes the inner product between two tensor trains
    """
    return jnp.sum(
        jnp.linalg.multi_dot([_tt_core_collapse(core_1, core_2) for core_1, core_2 in zip(tt_train_1, tt_train_2)]))


def tt_grad_inner_prod(tt_train_1: List[np.array], tt_train_2: List[np.array], gradient_core: np.array, idx):
    return jnp.sum(
        jnp.linalg.multi_dot(
            [_tt_core_collapse(core_1, gradient_core) if i == idx else _tt_core_collapse(core_1, core_2) for
             i, (core_1, core_2) in enumerate(zip(tt_train_1, tt_train_2))]))


def _tt_influence_core_collapse(core, idx):
    return sum([
        jnp.kron(core[(slice(None),) + i], core[(slice(None),) + i])
        for i in product(*[list(range(1, int(idx + k == 0) - 1, -1)) for k in range(len(core.shape) - 2)])
    ])


def tt_influence(tt_train: List[np.array], idx):
    """
    Returns the influence of an idx-index atom on a boolean function
    """

    return jnp.sum(
        jnp.linalg.multi_dot([_tt_influence_core_collapse(core, idx - i) for i, core in enumerate(tt_train)]))


def _tt_shared_influence_core_collapse(core, idx_1, idx_2):
    return sum([
        jnp.kron(core[(slice(None),) + i], core[(slice(None),) + i])
        for i in product(*[list(range(1, max(int(idx_1 + k == 0), int(idx_2 + k == 0)) - 1, -1)) for k in
                           range(len(core.shape) - 2)])
    ])


def tt_shared_influence(tt_train: List[np.array], idx_1, idx_2):
    """
    Returns the influence of an idx-index atom on a boolean function
    """

    return jnp.sum(
        jnp.linalg.multi_dot(
            [_tt_shared_influence_core_collapse(core, idx_1 - i, idx_2 - i) for i, core in enumerate(tt_train)]))


def _tt_phi_core(core: np.array):
    return sum([
        jnp.kron(
            jnp.expand_dims(core[(slice(None),) + i], list(range(1, 1 + len(i)))),
            phi(len(i) - 1)[(slice(None),) + sum(zip(i, [slice(None)] * len(i)), ())]
        ) for i in product(*([[0, 1]] * (len(core.shape) - 2)))])


def tt_bool_op(tt_train: List[np.array]) -> List[np.array]:
    """
    Produces the truth table result tensor
    """
    return [_tt_phi_core(core) for core in tt_train]


def _tt_phi_core_inv(core: np.array):
    return sum([
        jnp.kron(
            jnp.expand_dims(core[(slice(None),) + i], list(range(1, 1 + len(i)))),
            phi_inv(len(i) - 1)[(slice(None),) + sum(zip(i, [slice(None)] * len(i)), ())]
        ) for i in product(*([[0, 1]] * (len(core.shape) - 2)))])


def tt_bool_op_inv(tt_train: List[np.array]) -> List[np.array]:
    """
    Produces the truth table result tensor
    """
    return [_tt_phi_core_inv(core) for core in tt_train]


def _tt_measure_core(core: np.array, p, q):
    p_mat = np.array([[1, 1], [p, -q]], dtype=float).reshape(1, 2, 2, 1)
    print([p_mat[(slice(None),) + sum(zip(i, [slice(None)] * len(i)), ())] for i in product(*([[0, 1]] * (len(core.shape) - 2)))])
    return sum([
        jnp.kron(
            jnp.expand_dims(core[(slice(None),) + i], list(range(1, 1 + len(i)))),
            p_mat[(slice(None),) + sum(zip(i, [slice(None)] * len(i)), ())]
        ) for i in product(*([[0, 1]] * (len(core.shape) - 2)))])


def tt_measure(tt_train: List[np.array], likelihoods_true: np.array, likelihoods_false: np.array):
    """
    Returns a formula weighted by a measure defined via the likelihoods
    """
    return [_tt_measure_core(core, p, q) for core, p, q in zip(tt_train, likelihoods_true, likelihoods_false)]


def _tt_measure_core_inv(core: np.array, p, q):
    p_mat = (1/(p+q))*np.array([[q, 1], [p, -1]], dtype=float).reshape(1, 2, 2, 1)
    return sum([
        jnp.kron(
            jnp.expand_dims(core[(slice(None),) + i], list(range(1, 1 + len(i)))),
            p_mat[(slice(None),) + sum(zip(i, [slice(None)] * len(i)), ())]
        ) for i in product(*([[0, 1]] * (len(core.shape) - 2)))])


def tt_measure_inv(tt_train: List[np.array], likelihoods_true: np.array, likelihoods_false: np.array):
    """
    Returns a formula weighted by a measure defined via the likelihoods
    """
    return [_tt_measure_core_inv(core, p, q) for core, p, q in zip(tt_train, likelihoods_true, likelihoods_false)]


def boolean_criterion(dimension):
    one = tt_one(dimension)
    one[0] *= -1.0

    def criterion(tt_train):
        tt_train = tt_bool_op(tt_train)
        squared_Ttt_1 = tt_hadamard(tt_train, tt_train)
        squared_Ttt_1 = tt_rl_orthogonalize(squared_Ttt_1)
        minus_1_squared_Ttt_1 = tt_add(squared_Ttt_1, one)
        return tt_inner_prod(minus_1_squared_Ttt_1, minus_1_squared_Ttt_1)

    return criterion


def tt_extract_seq(tt_train, assignments):
    N = len(tt_train)
    tt_ttable = tt_rl_orthogonalize(tt_bool_op(tt_train))
    answer = [np.array([0.1, 0.9]).reshape(1, 2, 1) for _ in range(N)]  # Sum over tensor sums to 1
    indices = list(range(N))
    for i in assignments.keys():
        a = assignments[i]
        answer[i] = ((a - 1) / (-2) * np.array([0.0, 1.0]) + (1 + a) / 2 * np.array([1.0, 0.0])).reshape(1, 2, 1)
        indices.remove(i)
    termination_crit = 0.9 ** len(indices)  # if the 0.9s overlap on a truth value
    termination_crit -= 1 - termination_crit  # subtract all the other weights, i.e. 1 - true_entry (entries sum to 1)
    score = tt_inner_prod(tt_ttable, answer)
    while score < termination_crit:  # TODO: If we let it run it will converge to the memory mean
        improvements = -np.ones(N)
        buffer = deepcopy(answer)
        for j in indices:
            buffer[j] = np.array([0.9, 0.1]).reshape(1, 2, 1)
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
    tt_train_1 = tt_bool_op(tt_train_1)
    tt_train_2 = tt_bool_op(tt_train_2)
    tt_train_xnor = tt_hadamard(tt_train_1, tt_train_2)
    tt_train_xnor = tt_bool_op_inv(tt_train_xnor)
    return tt_rl_orthogonalize(tt_train_xnor)


def tt_xor(tt_train_1: List[np.array], tt_train_2: List[np.array]) -> List[np.array]:
    return tt_xnor(tt_neg(tt_train_1), tt_train_2)


def tt_and(tt_train_1: List[np.array], tt_train_2: List[np.array]) -> List[np.array]:
    tt_train_1 = tt_bool_op(tt_train_1)
    tt_train_2 = tt_bool_op(tt_train_2)
    tt_train_1[0] *= 0.5
    tt_mul = tt_hadamard(tt_train_1, tt_train_2)
    tt_train_2[0] *= 0.5
    half = tt_one(len(tt_train_1))
    half[0] *= -0.5
    tt_train_and = tt_add(tt_add(half, tt_mul), tt_add(tt_train_1, tt_train_2))
    tt_train_or = tt_bool_op_inv(tt_train_and)
    return tt_rl_orthogonalize(tt_train_or)


def tt_or(tt_train_1: List[np.array], tt_train_2: List[np.array]) -> List[np.array]:
    return tt_neg(tt_and(tt_neg(tt_train_1), tt_neg(tt_train_2)))


def tt_neg(tt_train: List[np.array]) -> List[np.array]:
    tt_train[0] *= -1
    return tt_train
