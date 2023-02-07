import numpy as np

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

    def find_feasible_hypothesis(self):
        tt_train, params = self._init_tt_train()
        criterion = boolean_criterion(self.dimension)
        indices = np.arange(self.dimension - 1)
        criterion_score = 1.0
        for _ in np.arange(int(params["lambda"]/0.02)):
            for idx in indices:
                tt_train = self._core_iteration(tt_train, params, idx)
            criterion_score = criterion(tt_train)
            params["mu"] += params["lr"]*criterion_score
            params["lambda"] -= 0.02
            params["lr"] *= 0.99
        while criterion_score > 1e-4:
            for idx in indices:
                tt_train = self._core_iteration(tt_train, params, idx)
            criterion_score = criterion(tt_train)
            params["mu"] += params["lr"]*criterion_score
            params["lr"] *= 0.99

        return tt_train

    def _core_iteration(self, tt_train, params, idx):
        B = jnp.einsum("abc, cde -> abde", tt_train[idx], tt_train[idx + 1])
        bonded_tt_train = tt_train[:idx] + [B] + tt_train[idx + 2:]
        B -= params["lr"] * self.gradient_functions[idx](*bonded_tt_train, params=params)
        B_part_1, B_part_2 = part_bond(B)
        tt_train = tt_train[:idx] + [B_part_1, B_part_2] + tt_train[idx + 2:]
        return tt_train

    def _boolean_criterion(self, idx):
        minus_one = tt_one_bonded(self.dimension, idx)
        minus_one[0] *= -1.0
        penalty = self._barrier(idx)

        def criterion_func(*tt_train, params):
            tt_train = tt_bool_op(tt_train)
            squared_Ttt_1 = tt_hadamard(tt_train, tt_train)
            minus_1_squared_Ttt_1 = tt_add(squared_Ttt_1, minus_one)
            return params['mu']*tt_inner_prod(minus_1_squared_Ttt_1, minus_1_squared_Ttt_1) - params['lambda']*penalty(tt_train, params['lambda'])

        return criterion_func

    def _barrier(self, idx):
        constraints = [c(idx) for c in self.constraints]

        def penalty(tt_train, shift):
            return sum([jnp.log(c(tt_train)+shift) for c in constraints])

        return penalty

    def _init_tt_train(self):
        tt_train = [2 * np.random.rand(1, 2, 1) - 1 for _ in range(self.dimension)]
        constraints = [c(-1) for c in self.constraints]
        params = {
            "lambda": 1 - min([c(tt_train) for c in constraints]),
            "mu": 1.0,
            "lr": 1e-2
        }
        return tt_train, params
