import numpy as np
from tt_op import *
from tt_op import _tt_train_kron
from copy import deepcopy
from utils import *
from optimiser import Minimiser



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




"""
def _boolean_criterion(tt_train):
    squared_Ttt_1 = tt_hadamard(tt_train, tt_train)
    minus_one = tt_one_bonded(5, 1)
    minus_one[0] *= -1
    minus_1_squared_Ttt_1 = tt_add(squared_Ttt_1, minus_one)
    return tt_inner_prod(minus_1_squared_Ttt_1, minus_1_squared_Ttt_1)
"""
"""
tt_1 = tt_svd(T_4)
tt_2 = tt_svd(T_4)
tt_2new = [tt_2[0], np.einsum("abc, cde -> abde", tt_2[1], tt_2[2]), tt_2[3], tt_2[4]]
tt_copy = deepcopy(tt_2new)
tt_crit = tt_bool_op(tt_2new)
parted_bond_1, parted_bond_2 = part_bond(tt_crit[1])
tt_crit = [tt_crit[0], parted_bond_1, parted_bond_2, tt_crit[2], tt_crit[3]]
print("hi", tt_inner_prod(tt_bool_op(tt_2), tt_bool_op(tt_2)))
print(tt_to_tensor(tt_crit))
print(tt_to_tensor(tt_bool_op(tt_2)))
"""
"""
tt_copy[0] *= -1
diff_tt = tt_add(tt_2new, tt_copy)
parted_bond_1, parted_bond_2 = part_bond(diff_tt[1])
diff_tt = [diff_tt[0], parted_bond_1, parted_bond_2, diff_tt[2], diff_tt[3]]
print(np.round(tt_to_tensor(diff_tt), decimals=5))
"""
"""
print([t.shape for t in tt_2new])
contracted_hadamard = tt_hadamard(tt_2new, tt_2new)
parted_bond_1, parted_bond_2 = part_bond(contracted_hadamard[1])
tt_2_parted = [contracted_hadamard[0], parted_bond_1, parted_bond_2, contracted_hadamard[2], contracted_hadamard[3]]
print(parted_bond_1.shape, parted_bond_2.shape)
print(np.round(tt_to_tensor(tt_2_parted)-tt_to_tensor(actual_hadamard), decimals=5))
"""
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
x = Atom(3, "x")
y = Atom(3, "y")
z = Atom(3, "z")
opt = Minimiser([
    exists_A_extending(x << y),
    all_A_not_extending(y & z)
], 3)
hypothesis = opt.find_feasible_hypothesis()
#print(np.round(tt_to_tensor(tt_bool_op(hypothesis)), decimals=5).reshape(-1, 1))
print(get_ANF([x, y, z], hypothesis))
print(np.round(tt_to_tensor(hypothesis), decimals=5))
print("Score:", boolean_criterion(hypothesis), tt_inner_prod(hypothesis, hypothesis))



