import numpy as np
from tt_op import *

T_1 = np.array([[[0, 1 / 2],
                 [1 / 2, 0]],
                [[1 / 2, 0],
                 [0, -1 / 2]]])

t_1 = lambda x, y, z: not (not (x and y) and not (x and z) and not (y and z))

T_2 = np.array([[[-3 / 4, 1 / 4],  # x and y and z
                 [1 / 4, 1 / 4]],
                [[1 / 4, 1 / 4],
                 [1 / 4, 1 / 4]]])

t_2 = lambda x, y, z: (x and y and z)

test_values = [[True, True, True],
               [True, False, True],
               [True, True, False],
               [True, False, False],
               [False, True, True],
               [False, False, True],
               [False, True, False],
               [False, False, False]]


def test_tt_svd():
    tts = tt_svd(T_1)
    retensor = tt_to_tensor(tts)
    assert np.sum(np.abs(T_1 - retensor)) < 1e-5


def test_tt_add():
    tt_1 = tt_svd(T_1)
    tt_2 = tt_svd(T_2)
    added_tt = tt_add(tt_1, tt_2)
    retensor = tt_to_tensor(added_tt)
    assert np.sum(np.abs(retensor - (T_1 + T_2))) < 1e-5


def test_tt_multiply():
    tt_1 = tt_svd(T_1)
    tt_2 = tt_svd(T_2)
    multiplied_tt = tt_hadamard(tt_1, tt_2)
    retensor = tt_to_tensor(multiplied_tt)
    assert np.sum(np.abs(retensor - (T_1 * T_2))) < 1e-5


def test_tt_inner_product():
    tt_1 = tt_svd(T_1)
    tt_2 = tt_svd(T_2)
    for test_value in test_values:
        bool_tt_train = bool_to_tt_train(test_value)
        true_eval_1 = t_1(*test_value)
        true_eval_2 = t_2(*test_value)
        tt_eval_1 = tt_inner_prod(tt_1, bool_tt_train).item()
        tt_eval_2 = tt_inner_prod(tt_2, bool_tt_train).item()
        assert np.abs(2 * float(true_eval_1) - 1 - tt_eval_1) < 1e-5
        assert np.abs(2 * float(true_eval_2) - 1 - tt_eval_2) < 1e-5


def test_tt_bool_op():
    tt_1 = tt_svd(T_1)
    Ttt_1 = tt_bool_op(tt_1)
    print(Ttt_1)
    print([t.shape for t in Ttt_1])
    print(tt_to_tensor(Ttt_1))
    squared_Ttt_1 = tt_hadamard(Ttt_1, Ttt_1)
    print(tt_to_tensor(squared_Ttt_1))

test_tt_bool_op()