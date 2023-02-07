from tt_op import *
from operators import D_func


class Minimiser:
    def __init__(self, constraints, dimension):
        self.dimension = dimension
        self.constraints = constraints
        self.gradient_functions = []
        for idx in range(dimension - 1):
            self.gradient_functions.append(
                partial_D(self._boolean_criterion(idx), idx)
            )
        constraint_functions = [c(-1) for c in self.constraints]
        self.penalty_function = lambda tt_train: sum([c(tt_train) for c in constraint_functions])
        self.lr = 1e-2

    def find_feasible_hypothesis(self):
        tt_train, aux, params = self._init_tt_train()
        while aux > 0:
            for idx in range(self.dimension - 1):
                tt_train, aux = self._core_iteration(tt_train, aux, params, idx)
            aux -= params["mu"] / (aux + self.penalty_function(tt_train))
            self.lr *= 0.99
            params["mu"] *= 0.99

        return tt_train, aux

    def _core_iteration(self, tt_train, aux, params, idx):
        B = np.einsum("abc, cde -> abde", tt_train[idx], tt_train[idx + 1])
        bonded_tt_train = tt_train[:idx] + [B] + tt_train[idx + 2:]
        B -= self.lr * self.gradient_functions[idx](*bonded_tt_train, aux=aux, params=params)
        B_part_1, B_part_2 = part_bond(B)
        tt_train = tt_train[:idx] + [B_part_1, B_part_2] + tt_train[idx + 2:]
        return tt_train, aux

    def _boolean_criterion(self, idx):
        minus_one = tt_one_bonded(self.dimension, idx)
        minus_one[0] *= -1.0
        penalty = self._barrier(idx)

        def criterion_func(*tt_train, aux, params):
            tt_train = tt_bool_op(tt_train)
            squared_Ttt_1 = tt_hadamard(tt_train, tt_train)
            minus_1_squared_Ttt_1 = tt_add(squared_Ttt_1, minus_one)
            return aux + (1/(2*params["mu"]))*tt_inner_prod(minus_1_squared_Ttt_1, minus_1_squared_Ttt_1) - params["mu"]*penalty(tt_train, aux)

        return criterion_func

    def _barrier(self, idx):
        constraints = [c(idx) for c in self.constraints]

        def penalty_func(tt_train, aux):
            return sum([jnp.log(c(tt_train)+aux) for c in constraints])

        return penalty_func

    def _init_tt_train(self):
        tt_train = [2 * np.random.rand(1, 2, 1) - 1 for _ in range(self.dimension)]
        constraints = [c(-1) for c in self.constraints]
        aux = 1 - min([c(tt_train) for c in constraints])
        params = {"mu": 1.0}
        return tt_train, aux, params
