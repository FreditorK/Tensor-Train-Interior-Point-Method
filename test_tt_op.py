import pytest
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


@pytest.mark.parametrize("tensor", [T_1, T_2])
def test_tt_svd(tensor):
    tts = tt_svd(tensor)
    retensor = tt_to_tensor(tts)
    assert np.sum(np.abs(tensor - retensor)) < 1e-5


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


@pytest.mark.parametrize("tensor_tuple", [(T_1, t_1), (T_2, t_2)])
def test_tt_inner_product(tensor_tuple):
    tensor, t = tensor_tuple
    tt = tt_svd(tensor)
    for test_value in test_values:
        bool_tt_train = bool_to_tt_train(test_value)
        true_eval = t(*test_value)
        tt_eval = tt_inner_prod(tt, bool_tt_train).item()
        assert np.abs(2 * float(true_eval) - 1 - tt_eval) < 1e-5


@pytest.mark.parametrize("tensor", [T_1, T_2])
def test_tt_bool_op(tensor):
    tt = tt_svd(tensor)
    Ttt = tt_bool_op(tt)
    squared_Ttt_1 = tt_hadamard(Ttt, Ttt)
    minus_one = ONE(len(Ttt))
    minus_one[0] *= -1
    minus_1_squared_Ttt_1 = tt_add(squared_Ttt_1, minus_one)
    result = tt_inner_prod(minus_1_squared_Ttt_1, minus_1_squared_Ttt_1)
    assert abs(result) < 1e-5


@pytest.mark.parametrize("tensor", [T_1, T_2])
def test_tt_orthogonolize(tensor):
    tt = tt_svd(tensor)
    tt_ortho = tt_rl_orthogonalize(tt)
    assert np.sum(np.abs(tt_to_tensor(tt_ortho) - tensor)) < 1e-5


@pytest.mark.parametrize("tensor", [T_1, T_2])
def test_tt_round(tensor):
    tt = tt_svd(tensor)
    tt_added = tt_add(tt, tt)
    tt_rounded = tt_round(tt_added)
    retensor = tt_to_tensor(tt_rounded)
    assert np.sum(np.abs(retensor - (tensor+tensor))) < 1e-5

tt_1 = tt_svd(T_1)
tt_2 = tt_svd(T_2)
tt_and_12 = tt_or(tt_1, tt_2)
retensor = tt_to_tensor(tt_and_12)
print(np.sum(retensor*retensor))
print(retensor)