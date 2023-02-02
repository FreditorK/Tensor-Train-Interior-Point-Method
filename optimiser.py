from tt_op import *


class Minimiser:
    def __init__(self, constraints, dimension):
        self.dimension = dimension
        self.constraints = constraints
        self.gradient_functions = []
        for idx in range(dimension - 1):
            self.gradient_functions.append(
                partial_D(self._boolean_criterion(idx), idx)
            )
        self.lr = 1e-2

    def find_feasible_hypothesis(self):
        tt_train = self._init_tt_train()
        for _ in range(500):
            for idx in range(self.dimension - 1):
                tt_train = self._core_iteration(tt_train, idx)
            self.lr *= 0.99

        return tt_train

    def _core_iteration(self, tt_train, idx):
        B = np.einsum("abc, cde -> abde", tt_train[idx], tt_train[idx + 1])
        bonded_tt_train = tt_train[:idx] + [B] + tt_train[idx + 2:]
        B -= self.lr * self.gradient_functions[idx](*bonded_tt_train)
        B_part_1, B_part_2 = part_bond(B)
        tt_train = tt_train[:idx] + [B_part_1, B_part_2] + tt_train[idx + 2:]
        return tt_train

    def _boolean_criterion(self, idx):
        one = tt_one_bonded(self.dimension, idx)
        one[0] *= -1.0

        def criterion_func(*tt_train):
            tt_train = tt_bool_op(tt_train)
            squared_Ttt_1 = tt_hadamard(tt_train, tt_train)
            minus_1_squared_Ttt_1 = tt_add(squared_Ttt_1, one)
            return tt_inner_prod(minus_1_squared_Ttt_1,
                                 minus_1_squared_Ttt_1)

        return criterion_func

    def _projection(self, gradient: np.array):
        pass

    def _init_tt_train(self):
        return [2 * np.random.rand(1, 2, 1) - 1 for _ in range(self.dimension)]
