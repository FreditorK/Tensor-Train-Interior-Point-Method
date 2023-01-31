import numpy as np
from tt_op import *
from tt_op import _tt_train_kron
from copy import deepcopy
from utils import *



T_1 = np.array([[[2, 1],
                 [3, 5]],
                [[9, 4],
                 [7, 6]]])

T_2 = np.array([[[0, 0],
                 [1, 0]],
                [[0, 0],
                 [0, 0]]])

T_3 = np.array([
    [
        [[-1/4, 0],
         [0, 3/4]],
        [[1/4, 0],
         [0, 1/4]]
    ],
    [
        [[1/4, 0],
         [0, 1/4]],
        [[-1/4, 0],
         [0, -1/4]]
    ]
])

T_4 = np.array([
    [
        [
            [[-5/8, 3/8],
             [0, 0]],
            [[1/8, 1/8],
             [0, 0]]
        ],
        [
            [[1/8, 1/8],
             [0, 0]],
            [[-1/8, -1/8],
             [0, 0]]
        ]
    ],
    [
        [
            [[3/8, 3/8],
             [0, 0]],
            [[1/8, 1/8],
             [0, 0]]
        ],
        [
            [[1/8, 1/8],
             [0, 0]],
            [[-1/8, -1/8],
             [0, 0]]
        ]
    ]
])

T_5 = np.array([
    [
        [
            [[1, 1],
             [0, 0]],
            [[1, 1],
             [1, 0]]
        ],
        [
            [[1, 1],
             [0, 0]],
            [[1, 1],
             [0, 0]]
        ]
    ],
    [
        [
            [[1, 1],
             [1, 0]],
            [[1, 1],
             [0, 0]]
        ],
        [
            [[1, 1],
             [0, 0]],
            [[1, 1],
             [0, 1]]
        ]
    ]
])


def fake_tt_train_kron(core_1: np.array, core_2: np.array) -> np.array:
    """
    For internal use: Computes the kronecker product between two TT-cores with appropriate dimensional
    expansion
    """
    layers = []
    core_shape_length = len(core_1.shape)
    axes = [i for i in range(1, core_shape_length-1)]
    for i in product(*[[0, 1] for _ in range(core_shape_length-2)]):
        idx = (slice(None), ) + i + (slice(None), )
        layers.append(np.kron(np.expand_dims(core_1[idx], axis=axes), np.expand_dims(core_2[idx], axis=axes)))
    return np.concatenate(layers, axis=1).reshape(tuple(ele1 * ele2 if i==0 or i==core_shape_length-1 else ele1 for i, (ele1, ele2) in enumerate(zip(core_1.shape, core_2.shape))))


def fake_tt_hadamard(tt_train_1: List[np.array], tt_train_2: List[np.array]) -> List[np.array]:
    """
    Computes the hadamard product/pointwise multiplication of two tensor trains
    """
    new_cores = []
    for core_1, core_2 in zip(tt_train_1, tt_train_2):
        new_cores.append(fake_tt_train_kron(core_1, core_2))
    return new_cores

def part_bond(core):
    shape = core.shape
    A = core.reshape(shape[0] * shape[1], -1)
    U, S, V_T = np.linalg.svd(A)
    non_sing_eig_idxs = np.nonzero(S)[0]
    S = S[non_sing_eig_idxs]
    next_rank = len(S)
    U = U[:, non_sing_eig_idxs]
    V_T = V_T[non_sing_eig_idxs, :]
    G_i = U.reshape(shape[0], shape[1], next_rank)
    G_ip1 = (np.diag(S) @ V_T).reshape(next_rank, shape[2], shape[-1])
    return G_i, G_ip1


tt_1 = tt_svd(T_4)
actual_hadamard = tt_hadamard(tt_1, tt_1)
print(np.round(tt_to_tensor(actual_hadamard), decimals=5))
tt_2 = tt_svd(T_4)
tt_2new = [tt_2[0], np.einsum("abc, cde -> abde", tt_2[1], tt_2[2]), tt_2[3], tt_2[4]]
print([t.shape for t in tt_2new])
contracted_hadamard = fake_tt_hadamard(tt_2new, tt_2new)
parted_bond_1, parted_bond_2 = part_bond(contracted_hadamard[1])
tt_2_parted = [contracted_hadamard[0], parted_bond_1, parted_bond_2, contracted_hadamard[2], contracted_hadamard[3]]
print(parted_bond_1.shape, parted_bond_2.shape)
print(np.round(tt_to_tensor(tt_2_parted)-tt_to_tensor(actual_hadamard), decimals=5))
#tt_2 = tt_svd(T_2)
#print(T_4.shape)
#print(np.sum(T_4*T_4))
#tt_3 = tt_svd(T_5)
#print([t.shape for t in tt_3])
#print(np.round(tt_to_tensor(tt_3), decimals=4))
#rounded = tt_round(tt_3)
#print([t.shape for t in rounded])
#print([np.round(t, decimals=4) for t in tt_3])
#print(np.round(tt_to_tensor(rounded), decimals=4))

#tt_1_copy = deepcopy(tt_1)
#new_tt = tt_xor(tt_1, tt_2)
#print(new_tt)
#print([t.shape for t in new_tt])
#print(tt_to_tensor(new_tt))
#print(T_3.shape)
#print(tt_svd(T_3))
#print(tt_to_tensor(tt_3))
#print(tt_to_tensor(ONE(3)))

#x = Atom(2, "x")
#y = Atom(2, "y")
#z = Atom(3, "z")
#print(tt_to_tensor(y.to_tt_train()))
#formula = (x & y)# ^ z
#print(formula)
#tt_formula = formula.to_tt_train()
#print(tt_formula)
#print(tt_to_tensor(tt_formula))
#print(tt_leading_entry(tt_formula))
#print(tt_to_tensor(tt_formula))

