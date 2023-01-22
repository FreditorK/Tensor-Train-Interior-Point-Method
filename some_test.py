import numpy as np
from tt_op import *
from tt_op import _tt_train_kron
from copy import deepcopy


def fundamental_tt_train(idx, dim):
    tt_train = [np.zeros((1, 2, 2))] + [np.zeros((2, 2, 2))]*(dim-2) + [np.zeros((2, 2, 1))]
    multi_idx = [0]*dim
    multi_idx[idx] = 1
    for i in range(dim):
        tt_train[i][0, multi_idx[i], 0] = 1
    return tt_train

def tensor_xor(tensor):
    pass

# print(tt_to_tensor(fundamental_tt_train(0, 3)))

T_1 = np.array([[[2, 1],
                 [3, 5]],
                [[9, 4],
                 [7, 6]]])

T_2 = np.array([[[0, 0],
                 [1, 0]],
                [[0, 0],
                 [0, 0]]])

T_3 = np.array([[[0, 0],
                 [0, 1]],
                [[0, 0],
                 [0, 0]]])

#print(np.kron(T_2, T_3))
tt_1 = tt_svd(T_1)
#print(tt_1)
tt_2 = tt_svd(T_2)
#print(tt_2)
tt_3 = tt_svd(T_3)
#print(tt_3)
#print(np.kron(T_1.reshape(2, 2, 2, 1, 1, 1), T_1.reshape(1, 1, 1, 2, 2, 2))) # somehow subcomponents need to be permuted with something like [[0, 1], [1, 0]]
"""
tt_1[0] = np.array([[[1., 0.],
                     [0., 1.]]])
tt_1[1] = np.array([[[1., 0.],
                     [0., 1.]],
                    [[0., 0.],
                     [0., 0.]]])
"""
#tt_1[2][:, [0, 1], :] = tt_1[2][:, [1, 0], :]
#tt_1[1][:, [0, 1], :] = tt_1[2][:, [1, 0], :]
#print(np.einsum("i, abc -> abc", np.array([1, 0]), T_1))# + np.einsum("i, abc -> aic", np.array([0, 1]), T_1))

I_MATRIX = np.array([[1, 0],
                     [0, 1]], dtype=float).reshape(1, 2, 2, 1)

def tt_experimental_op(tt_train_1: List[np.array], tt_train_2: List[np.array]) -> List[np.array]:
    """
    Produces the truth table result tensor
    """
    new_cores = []
    for idx, (core_1, core_2) in enumerate(zip(tt_train_1, tt_train_2)):
        new_core = np.kron(core_1[:, None, 0, :], I_MATRIX[:, 0, :, :] @ core_2[:, None, 0, :]) + np.kron(core_1[:, None, 1, :], I_MATRIX[:, 1, :, :] @ core_2[:, None, 1, :])
        new_cores.append(new_core)
    return new_cores

#print(tt_to_tensor(tt_1))
tt_1_copy = deepcopy(tt_1)
#tt_1_copy[2][:, [0, 1], :] = tt_1_copy[2][:, [1, 0], :]
tt_1_copy[2] = np.kron(np.array([[0], [1]]), tt_1_copy[2][:, None, 0, :]) + np.kron(np.array([[1], [0]]), tt_1_copy[2][:, None, 1, :])
#print(tt_1[2])
#print(tt_1_copy[2])
print(tt_to_tensor(tt_1), tt_to_tensor(tt_1_copy))
#tt_1 = tt_experimental_op(tt_1, tt_1_copy)#[tt_1[0], tt_1[1], np.array([[[1, 0], [0, 1]], [[0, 1], [1, 0]]]), tt_1[2]]
#inner_pro = tt_inner_prod(tt_1, tt_1_copy)
#print(inner_pro)
#print(tt_to_tensor(tt_1))