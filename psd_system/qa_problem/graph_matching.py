import copy
import sys
import os


sys.path.append(os.getcwd() + '/../../')

from dataclasses import dataclass
from src.tt_ops import *
from src.tt_ops import _tt_core_collapse, tt_random_gaussian
from psd_system.graph_plotting import *
from src.tt_ipm import tt_ipm, _tt_get_block
import time


@dataclass
class Config:
    seed = 4
    ranks = [3]


def _core_Q_diag_mP(c, i):
    mask = np.zeros((c.shape[0], 4, *c.shape[1:]))
    if i == 0:
        mask[:, 1] = c
    else:
        mask[:, 0, 0, 0] = c[:, 0, 0]
        mask[:, 1, 0, 1] = c[:, 0, 1]
        mask[:, 2, 1, 0] = c[:, 1, 0]
        mask[:, 2, 1, 1] = c[:, 1, 1]
    return mask

def _core_Q_diag_mPT(c, i):
    mask = np.zeros((c.shape[0], 4, *c.shape[1:]))
    if i == 0:
        mask[:, 2] = c
    else:
        mask[:, 0, 0, 0] = c[:, 0, 0]
        mask[:, 1, 0, 1] = c[:, 0, 1]
        mask[:, 1, 1, 1] = c[:, 1, 1]
        mask[:, 2, 1, 0] = c[:, 1, 0]
    return mask

def _core_Q_diag_mPPT(c, i):
    mask = np.zeros((c.shape[0], 4, *c.shape[1:]))
    if i == 0:
        mask[:, 0] = c
    else:
        mask[:, 0, 0, 0] = c[:, 0, 0]
        mask[:, 3, 0, 1] = c[:, 0, 1]
        mask[:, 3, 1, 0] = c[:, 1, 0]
        mask[:, 3, 1, 1] = c[:, 1, 1]
    return mask


def tt_Q_m_P_op(dim):
    v_matrix_1 = [np.array([[1, 0], [0, 0]]).reshape(1, 2, 2, 1)] + tt_identity(dim)
    v_matrix_2 = [np.array([[0.0, -1.0], [0.0, 0.0]]).reshape(1, 2, 2, 1)] + [
        np.array([[1.0, 0.0], [1.0, 0.0]]).reshape(1, 2, 2, 1) for _ in range(dim)]
    matrix_tt = tt_rank_reduce(tt_add(v_matrix_1, v_matrix_2))
    custom_op_1 = [_core_Q_diag_mP(c, i) for i, c in enumerate(matrix_tt)]
    custom_op_2 = [_core_Q_diag_mPT(c, i) for i, c in enumerate(tt_transpose(matrix_tt))]
    custom_op_3 = [_core_Q_diag_mPPT(c, i) for i, c in enumerate(tt_scale(0.5, tt_add(matrix_tt, tt_transpose(matrix_tt))))]
    custom_op = tt_add(custom_op_1, custom_op_2)
    return tt_rank_reduce(tt_add(custom_op, custom_op_3))


def _core_diag_block_sum():
    mask = np.zeros((4, 2, 2))
    mask[0, :, :] = np.eye(2)
    mask[1, :, :] = np.zeros((2, 2))
    mask[2, :, :] = np.zeros((2, 2))
    mask[3, :, :] = np.eye(2)
    return mask.reshape(1, 4, 2, 2, 1)


def _core_diag_block_sum_block():
    mask = np.zeros((4, 2, 2))
    mask[0, :, :] = np.array([[1, 0], [0, 0]])
    mask[1, :, :] = np.array([[0, 1], [0, 0]])
    mask[2, :, :] = np.array([[0, 0], [1, 0]])
    mask[3, :, :] = np.array([[0, 0], [0, 1]])
    return mask.reshape(1, 4, 2, 2, 1)


def tt_diag_block_sum_linear_op(block_size, dim):
    """
    Linear operator that sums the block diagonal of a TT-matrix and allocates it to the block diagonals,
    where the blocks are of size 2^block_size x 2^block_size and the matrix is 2^dim x 2^dim
    """
    return ([_core_diag_block_sum() for _ in range(dim - block_size)]
            + [_core_diag_block_sum_block() for _ in range(block_size)])


def _core_partial_J_trace_middle_2(c):
    mask_1 = np.zeros((c.shape[0], 4, *c.shape[1:]))
    mask_1[:, 0] = c
    mask_2 = np.zeros((c.shape[0], 4, *c.shape[1:]))
    #mask_2[:, 1] = c
    mask_3 = np.zeros((c.shape[0], 4, *c.shape[1:]))
    #mask_3[:, 2] = c
    mask_4 = np.zeros((c.shape[0], 4, *c.shape[1:]))
    #mask_4[:, 3] = c
    return mask_1#np.concatenate((mask_1, mask_2, mask_3, mask_4), axis=0)

def _core_partial_J_trace_middle_1(c):
    mask_1 = np.zeros((c.shape[0], 4, *c.shape[1:]))
    mask_1[:, 0, 0, 0] = c[:, 0, 0]
    #mask_2 = np.zeros((c.shape[0], 4, *c.shape[1:]))
    #mask_2[:, 0, 0, 1] = c[:, 0, 1]
    #mask_3 = np.zeros((c.shape[0], 4, *c.shape[1:]))
    #mask_3[:, 0, 1, 0] = c[:, 1, 0]
    #mask_4 = np.zeros((c.shape[0], 4, *c.shape[1:]))
    #mask_4[:, 0, 1, 1] = c[:, 1, 1]
    return mask_1 #np.concatenate((mask_1, mask_2, mask_3, mask_4), axis=-1)


def _core_partial_J_trace(c):
    mask = np.zeros((c.shape[0], 4, *c.shape[1:]))
    mask[:, 0] = c
    return mask


def tt_partial_J_trace_op(block_size, dim):
    matrix_tt = [np.ones((1, 2, 2, 1)) for _ in range(dim)]
    first_cores = tt_mask_to_linear_op(matrix_tt[:dim-block_size-1])
    middle_cores = [_core_partial_J_trace_middle_1(matrix_tt[dim-block_size-1]), _core_partial_J_trace_middle_2(matrix_tt[dim-block_size])]
    last_cores = [_core_partial_J_trace(c) for c in matrix_tt[dim-block_size+1:]]
    return first_cores + middle_cores + last_cores

def _core_partial_J_trace_adjoint(c):
    mask = np.zeros((c.shape[0], 4, *c.shape[1:]))
    mask[:, 0] = c
    mask[:, 1] = c
    mask[:, 2] = c
    mask[:, 3] = c
    return mask


def tt_partial_J_trace_op_adjoint(block_size, dim):
    matrix_tt = [np.ones((1, 2, 2, 1)) for _ in range(dim-block_size)] + [np.array([[1.0, 0.0], [0.0, 0.0]]).reshape(1, 2, 2, 1) for _ in range(block_size)]
    return tt_mask_to_linear_op(matrix_tt[:-block_size]) + [_core_partial_J_trace_adjoint(c) for c in matrix_tt[-block_size:]]


def _core_partial_trace(c):
    mask = np.zeros((c.shape[0], 4, *c.shape[1:]))
    mask[:, 1, 0, 1] = c[:, 0, 1]
    mask[:, 1, 1, 0] = c[:, 0, 1]
    mask[:, 2, 1, 0] = c[:, 1, 0]
    mask[:, 2, 0, 1] = c[:, 0, 1]
    mask[:, 3, 0, 0] = c[:, 0, 0]
    mask[:, 3, 1, 1] = c[:, 1, 1]
    return mask

def tt_partial_trace_op(block_size, dim, off_diagonal=True):
    matrix_tt = [np.ones((1, 2, 2, 1)) for _ in range(dim-block_size)] + tt_identity(block_size)
    identity = tt_identity(dim)
    if off_diagonal:
        matrix_tt = tt_rank_reduce(tt_sub(matrix_tt, identity))
    return tt_mask_to_linear_op(matrix_tt[:-block_size]) + [_core_partial_trace(c) for c in matrix_tt[-block_size:]]


def _core_partial_trace_adjoint(c):
    mask = np.zeros((c.shape[0], 4, *c.shape[1:]))
    mask[:, 0, 0, 0] = c[:, 0, 0]
    mask[:, 0, 1, 1] = c[:, 1, 1]
    mask[:, 1, 0, 1] = c[:, 0, 1]
    mask[:, 1, 1, 0] = c[:, 0, 1]
    mask[:, 2, 1, 0] = c[:, 1, 0]
    mask[:, 2, 0, 1] = c[:, 0, 1]
    mask[:, 3, 0, 0] = c[:, 0, 0]
    mask[:, 3, 1, 1] = c[:, 1, 1]
    return mask

def tt_partial_trace_op_adjoint(block_size, dim, off_diagonal=True):
    matrix_tt = [np.ones((1, 2, 2, 1)) for _ in range(dim-block_size)] + [np.array([[0.0, 0.0], [0.0, 1.0]]).reshape(1, 2, 2, 1) for _ in range(block_size)]
    identity = tt_identity(dim-block_size) + [np.array([[0.0, 0.0], [0.0, 1.0]]).reshape(1, 2, 2, 1) for _ in range(block_size)]
    if off_diagonal:
        matrix_tt = tt_rank_reduce(tt_sub(matrix_tt, identity))
    return tt_mask_to_linear_op(matrix_tt[:-block_size]) + [_core_partial_trace_adjoint(c) for c in matrix_tt[-block_size:]]


def _core_ds_P(limit, c, i):
    mask = np.zeros((c.shape[0], 4, *c.shape[1:]))
    if i > limit:
        mask[:, 0, 0, 0] = c[:, 0, 0]
        mask[:, 0, 1, 0] = c[:, 1, 0]
        mask[:, 1, 0, 1] = c[:, 0, 1]
        mask[:, 2, 0, 0] = c[:, 0, 0]
        mask[:, 2, 1, 0] = c[:, 1, 0]
        mask[:, 3, 1, 1] = c[:, 1, 1]
    else:
        mask[:, 0, 0, 0] = c[:, 0, 0]
        mask[:, 1, 0, 1] = c[:, 0, 1]
        mask[:, 2, 1, 0] = c[:, 1, 0]
        mask[:, 3, 1, 1] = c[:, 1, 1]
    return mask

def _core_ds_PT(limit, c, i):
    mask = np.zeros((c.shape[0], 4, *c.shape[1:]))
    if i > limit:
        mask[:, 0, 0, 0] = c[:, 0, 0]
        mask[:, 0, 0, 1] = c[:, 0, 1]
        mask[:, 1, 0, 0] = c[:, 0, 0]
        mask[:, 1, 0, 1] = c[:, 0, 1]
        mask[:, 2, 1, 0] = c[:, 1, 0]
        mask[:, 3, 1, 1] = c[:, 1, 1]
    else:
        mask[:, 0, 0, 0] = c[:, 0, 0]
        mask[:, 1, 0, 1] = c[:, 0, 1]
        mask[:, 2, 1, 0] = c[:, 1, 0]
        mask[:, 3, 1, 1] = c[:, 1, 1]
    return mask


def tt_DS_op(block_size, dim):
    matrix_tt = [np.array([[0.0, 1.0], [0.0, 0.0]]).reshape(1, 2, 2, 1)] + [
        np.array([[1.0, 0.0], [1.0, 0.0]]).reshape(1, 2, 2, 1) for _ in range(dim)]
    custom_op_1 = [_core_ds_P(dim-block_size, c, i) for i, c in enumerate(matrix_tt)]
    custom_op_2 = [_core_ds_PT(dim-block_size, c, i) for i, c in enumerate(tt_transpose(matrix_tt))]
    custom_op = tt_add(custom_op_1, custom_op_2)
    return custom_op

def tt_DS_bias(dim):
    matrix_tt = [np.array([[0.0, 1.0], [0.0, 0.0]]).reshape(1, 2, 2, 1)] + [np.array([[1.0, 0.0], [1.0, 0.0]]).reshape(1, 2, 2, 1) for _ in range(dim-1)]
    return tt_rank_reduce(tt_add(matrix_tt, tt_transpose(matrix_tt)))

def tt_padding_op(dim):
    matrix_tt_1 = [np.array([[0.0, 1.0], [1.0, 1.0]]).reshape(1, 2, 2, 1)] + [np.ones((1, 2, 2, 1)) for _ in range(dim-1)]
    matrix_tt_2 = [np.array([[0.0, 0.0], [1.0, 0.0]]).reshape(1, 2, 2, 1)] +  [np.array([[1.0, 1.0], [0.0, 0.0]]).reshape(1, 2, 2, 1) for _ in
                                                                              range(dim - 1)]
    matrix_tt_3 = [np.array([[0.0, 1.0], [0.0, 0.0]]).reshape(1, 2, 2, 1)] + [
        np.array([[1.0, 0.0], [1.0, 0.0]]).reshape(1, 2, 2, 1) for _ in
        range(dim - 1)]
    matrix_tt = tt_rank_reduce(tt_sub(tt_sub(matrix_tt_1, matrix_tt_2), matrix_tt_3))
    return tt_mask_to_linear_op(matrix_tt)

def tt_padding_op_adjoint(dim):
    matrix_tt_1 = [np.array([[0.0, 1.0], [1.0, 1.0]]).reshape(1, 2, 2, 1)] + [np.ones((1, 2, 2, 1)) for _ in range(dim-1)]
    matrix_tt_2 = [np.array([[0.0, 0.0], [1.0, 0.0]]).reshape(1, 2, 2, 1)] +  [np.array([[1.0, 1.0], [0.0, 0.0]]).reshape(1, 2, 2, 1) for _ in
                                                                              range(dim - 1)]
    matrix_tt_3 = [np.array([[0.0, 1.0], [0.0, 0.0]]).reshape(1, 2, 2, 1)] + [
        np.array([[1.0, 0.0], [1.0, 0.0]]).reshape(1, 2, 2, 1) for _ in
        range(dim - 1)]
    matrix_tt = tt_rank_reduce(tt_sub(tt_sub(matrix_tt_1, matrix_tt_2), matrix_tt_3))
    return tt_mask_to_linear_op_adjoint(matrix_tt)


if __name__ == "__main__":
    print("Creating Problem...")
    """
        [Q   P  0 ]
    X = [P^T 1  0 ]
        [0   0  I ]
    e.g.
        [Q_11 Q_12 Q_13 Q_14 | P_11 |   0    0    0]
        [Q_21 Q_22 Q_23 Q_24 | P_21 |   0    0    0]
        [Q_31 Q_32 Q_33 Q_34 | P_12 |   0    0    0]
        [Q_41 Q_42 Q_43 Q_44 | P_22 |   0    0    0]
    X = [------------------------------------------]
        [P_11 P_21 P_12 P_22 |    1 |   0    0    0]
        [------------------------------------------]
        [   0    0    0    0 |    0 |   1    0    0]
        [   0    0    0    0 |    0 |   0    1    0]
        [   0    0    0    0 |    0 |   0    0    1]
        
        
        [ 5  5  | 0  0  | 7 | P P P]
        [ 5  5  | 0  4  | 7 | P P P]
        [--------------------------]
        [ 0  0  | 6  6  | 7 | P P P]
        [ 0  4  | 6  6  | 7 | P P P]
    Y = [--------------------------]
        [ 8r 8c | 8r 8c | P | P P P]
        [--------------------------]
        [ P  P  | P  P  | P | P P P]
        [ P  P  | P  P  | P | P P P]
        [ P  P  | P  P  | P | P P P]
    """
    q_op_prefix = np.zeros((1, 4, 2, 2, 1))
    q_op_prefix[0, 0, 0, 0, 0] = 1
    padding_op_prefix = np.zeros((1, 4, 2, 2, 1))
    padding_op_prefix[0, 3, 1, 1, 0] = 1
    q_bias_prefix = np.zeros((1, 2, 2, 1))
    q_bias_prefix[0, 0, 0, 0] = 1
    padding_bias_prefix = np.zeros((1, 2, 2, 1))
    padding_bias_prefix[0, 1, 1, 0] = 1
    np.random.seed(Config.seed)
    #graph_A = tt_random_graph(Config.ranks)
    #G_A = tt_scale(0.5, tt_add(graph_A, tt_one_matrix(len(Config.ranks) + 1)))
    #G_A = tt_rank_reduce(G_A)

    #print(np.round(tt_matrix_to_matrix(G_A), decimals=2))

    #graph_B = tt_random_graph(Config.ranks)
    #G_B = tt_scale(0.5, tt_add(graph_B, tt_one_matrix(len(Config.ranks) + 1)))
    #G_B = tt_rank_reduce(G_B)
    #print(np.round(tt_matrix_to_matrix(G_B), decimals=2))

    test_graph = tt_random_gaussian([3, 3, 3], shape=(2, 2)) #tt_random_graph([3])
    #test_graph = tt_scale(0.5, tt_add(test_graph, tt_one_matrix(len(Config.ranks) + 1)))
    G = tt_one_matrix(5) #tt_rank_reduce(test_graph) + [np.array([[0.9, 0.8], [0.7, 0.6]]).reshape(1, 2, 2, 1)]
    np.set_printoptions(linewidth=600, threshold=np.inf, precision=4, suppress=True)
    print(np.round(tt_matrix_to_matrix(G), decimals=2))

    n = 2 # 2^1
    n_sq = 4
    # Equality Operator
    # IV
    #partial_tr_op = [q_op_prefix] + tt_partial_trace_op(n, n_sq)
    #partial_tr_op_adjoint = [q_op_prefix] + tt_partial_trace_op_adjoint(n, n_sq)
    #partial_tr_op_bias = tt_zero_matrix(n_sq+1)
    # ---
    # V
    partial_tr_J_op = [q_op_prefix] + tt_partial_J_trace_op(n, n_sq)
    #partial_tr_J_op_adjoint = [q_op_prefix] + tt_partial_J_trace_op_adjoint(n, n_sq)
    #partial_tr_J_op_bias = [q_bias_prefix] + tt_one_matrix(n_sq)
    print(np.round(tt_matrix_to_matrix(tt_mat(tt_linear_op(partial_tr_J_op, G), shape=(2, 2))), decimals=2))
    # ---
    # VI
    #diag_block_sum_op = [q_op_prefix] + tt_diag_block_sum_linear_op(n, n_sq)
    #diag_block_sum_op_bias = [q_bias_prefix] + tt_identity(n_sq)
    # ---
    # VII
    #Q_m_P_op = tt_Q_m_P_op(n_sq)
    #Q_m_P_op_bias = tt_zero_matrix(n_sq+1)
    # ---
    # VIII
    #DS_op = tt_DS_op(n, n_sq)
    #DS_op_bias = tt_DS_bias(n_sq+1)
    # ---
    # IX
    #padding_op = tt_padding_op(n_sq+1)
    #padding_op_adjoint = tt_padding_op_adjoint(n_sq+1)
    #padding_op_bias = [padding_bias_prefix] + tt_identity(n_sq)
    # ---
    # Inequality Operator
    # X
    #Q_ineq_op = [q_op_prefix] + tt_mask_to_linear_op(tt_one_matrix(n_sq))
    #Q_ineq_bias = tt_zero_matrix(n_sq+1)
    # ---
