from copy import deepcopy

import numpy as np

from tt_op import *
from operators import D_func
from utils import ConstraintSpace

#np.random.seed(7)


class Minimiser:
    def __init__(self, constraints: ConstraintSpace, dimension):
        self.dimension = dimension
        self.const_space = constraints
        self.inequality_constraints = [lambda _: lambda h: 1 - tt_leading_entry(h)] + constraints._return_inequality_constraints()
        self.gradient_functions = []
        for idx in range(dimension - 1):
            self.gradient_functions.append(
                partial_D(self._barrier(idx), idx)
            )
        constraint_functions_iq = [c(-1) for c in self.inequality_constraints]
        self.penalty_function_iq = lambda tt_train: np.abs(min([c(tt_train) for c in constraint_functions_iq] + [0]))
        self.complete_gradient = D_func(self.criterion(dimension))

    def criterion(self, dimension):
        one = tt_one(dimension)
        one[0] *= -1.0

        @jax.jit
        def criterion(tt_train, params):
            tt_train = tt_bool_op(tt_train)
            squared_Ttt_1 = tt_hadamard(tt_train, tt_train)
            minus_1_squared_Ttt_1 = tt_add(squared_Ttt_1, one)
            return tt_inner_prod(minus_1_squared_Ttt_1, minus_1_squared_Ttt_1)

        return criterion

    def find_feasible_hypothesis(self):
        tt_train, params = self._init_tt_train()
        """
        criterion = boolean_criterion(self.dimension)
        indices = np.arange(self.dimension - 1)
        prev_max_violation = -np.inf
        while params["lambda"] > 0:
            for idx in indices:
                tt_train = self._bonded_iteration(tt_train, params, idx)
            max_violation = self.penalty_function_iq(tt_train)
            print(f"Current violation: {max_violation} \r", end="")
            params["lambda"] = max(params["lambda"] - 0.05, max_violation)
            if prev_max_violation < max_violation:
                params["lr"] *= 0.99
            prev_max_violation = max_violation
        params["lambda"] = 0
        params["lr"] *= 0.1
        print("Barrier feasible!", flush=True)
        prev_criterion_score = np.inf
        criterion_score = np.inf
        while criterion_score > 1e-4:
            for idx in range(self.dimension-1):
                tt_train = self._iteration(tt_train,  params, idx)
            tt_train = self.const_space.project(tt_train)
            criterion_score = criterion(tt_train)
            if criterion_score > prev_criterion_score:
                params["lr"] *= 0.995
            prev_criterion_score = criterion_score
            print(f"Current violation: {criterion_score} \r", end="")
        print("\n", flush=True)
        """
        return tt_train

    def _bonded_iteration(self, tt_train, params, idx):
        B = jnp.einsum("abc, cde -> abde", tt_train[idx], tt_train[idx + 1])
        bonded_tt_train = tt_train[:idx] + [B] + tt_train[idx + 2:]
        grad = self.gradient_functions[idx](*bonded_tt_train, params=params)
        B -= params["lr"] *(grad - tt_grad_inner_prod(bonded_tt_train, bonded_tt_train, grad, idx)*B)
        B_part_1, B_part_2 = part_bond(B)
        tt_train = tt_train[:idx] + [B_part_1, B_part_2] + tt_train[idx + 2:]
        tt_train[idx] = tt_train[idx] / jnp.sqrt(tt_inner_prod(tt_train, tt_train))
        return tt_train

    def _iteration(self, tt_train, params, idx):
        gradient = self.complete_gradient(tt_train, params)[idx]
        tt_train[idx] -= params["lr"] * (
            gradient - tt_grad_inner_prod(tt_train, tt_train, gradient, idx) * tt_train[idx])
        tt_train[idx] = tt_train[idx] / jnp.sqrt(tt_inner_prod(tt_train, tt_train))
        return tt_train

    def _barrier(self, idx):
        inequality_constraints = [c(idx) for c in self.inequality_constraints]

        def penalty(*tt_train, params):
            return - sum([jnp.log(c(tt_train) + params["lambda"]) for c in inequality_constraints])

        return penalty

    def _init_tt_train(self):
        tt_train = [2 * np.random.rand(1, 2, 1) - 1 for _ in range(self.dimension)]
        tt_train[0] = tt_train[0] / np.sqrt(tt_inner_prod(tt_train, tt_train))
        tt_train = tt_rl_orthogonalize(tt_train)
        print("Before", [c(tt_train) for c in self.const_space.eq_constraints])
        tt_train = self.const_space.project(tt_train)
        print("After", [c(tt_train) for c in self.const_space.eq_constraints])
        params = {
            "lambda": 1 - self.penalty_function_iq(tt_train),
            "lr": 1e-2
        }
        return tt_train, params
