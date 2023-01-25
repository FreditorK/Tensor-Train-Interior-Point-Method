import numpy as np
from tt_op import *
from tt_op import _tt_train_kron
from copy import deepcopy


def fundamental_tt_train(idx, dim):
    tt_train = [np.zeros((1, 2, 2))] + [np.zeros((2, 2, 2))] * (dim - 2) + [np.zeros((2, 2, 1))]
    multi_idx = [0] * dim
    multi_idx[idx] = 1
    for i in range(dim):
        tt_train[i][0, multi_idx[i], 0] = 1
    return tt_train


T_1 = np.array([[[2, 1],
                 [3, 5]],
                [[9, 4],
                 [7, 6]]])

T_2 = np.array([[[0, 0],
                 [1, 0]],
                [[0, 0],
                 [0, 0]]])

T_3 = np.array([[[1, 0],
                 [0, 0]],
                [[0, 0],
                 [0, 0]]])

tt_1 = tt_svd(T_1)
tt_2 = tt_svd(T_2)
tt_3 = tt_svd(T_3)

tt_1_copy = deepcopy(tt_1)
new_tt = tt_xor(tt_1, tt_2)
print(new_tt)
print([t.shape for t in new_tt])
print(tt_to_tensor(new_tt))
print(tt_svd(T_3))
