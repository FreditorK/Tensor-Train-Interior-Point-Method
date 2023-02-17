from copy import deepcopy

import numpy as np

from tt_op import *
from operators import D_func
from utils import ConstraintSpace

#np.random.seed(7)


class Minimiser:
    def __init__(self, constraints: ConstraintSpace, dimension):
        self.dimension = dimension
        self.equality_constraints = constraints._return_equality_constraints()
        self.inequality_constraints = [lambda _: lambda h: 1 - tt_leading_entry(h)] + constraints._return_inequality_constraints()
        self.gradient_functions = []
        for idx in range(dimension - 1):
            self.gradient_functions.append(
                partial_D(self._barrier(idx), idx)
            )
        constraint_functions_eq = [c(-1) for c in self.equality_constraints]
        constraint_functions_iq = [c(-1) for c in self.inequality_constraints]
        self.penalty_function_eq = lambda tt_train: np.amax(np.abs([c(tt_train) for c in constraint_functions_eq] + [0]))
        self.penalty_function_iq = lambda tt_train: np.abs(min([c(tt_train) for c in constraint_functions_iq] + [0]))
        self.complete_gradient = D_func(boolean_criterion(dimension))

    def find_feasible_hypothesis(self):
        tt_train, params = self._init_tt_train()
        criterion = boolean_criterion(self.dimension)
        indices = np.arange(self.dimension - 1)
        prev_max_violation = -np.inf
        while params["lambda"] > 0:
            for idx in indices:
                tt_train = self._core_iteration(tt_train, params, idx)
            max_violation = self.penalty_function_iq(tt_train)
            print(f"Current violation: {max_violation} \r", end="")
            params["lambda"] = max(params["lambda"] - 0.05, max_violation)
            if prev_max_violation > max_violation:
                params["lr"] *= 0.99
        params["lambda"] = 0
        params["beta"] = 1.0
        print("Barrier feasible!", flush=True)
        while params["mu"] > 1e-3:
            for idx in indices:
                tt_train = self._core_iteration(tt_train, params, idx)
            max_violation = self.penalty_function_eq(tt_train)
            print(f"Current violation: {max_violation} \r", end="")
            params["mu"] = max(params["mu"]-0.025, max_violation)
            if prev_max_violation > max_violation:
                params["lr"] *= 0.99
        print("Feasible point found.", flush=True)
        params["mu"] = 0
        params["lr"] *= 0.1
        prev_criterion_score = np.inf
        criterion_score = 1.0
        print(tt_to_tensor(tt_bool_op(tt_train)))
        tt_save = deepcopy(tt_train)
        while criterion_score > 1e-4:
            gradient = self.complete_gradient(tt_train)
            tt_train = [t - params["lr"] * gradient[i] for i, t in enumerate(tt_train)]
            criterion_score = criterion(tt_train)
            print(f"Current violation: {criterion_score} \r", end="")
            if criterion_score >= prev_criterion_score:
                params["lr"] *= 0.9
            prev_criterion_score = criterion_score
        print("\n", flush=True)
        print("Inner", (1-tt_inner_prod(tt_save, tt_train))/2, flush=True)
        return tt_rank_reduce(tt_train)

    def _core_iteration(self, tt_train, params, idx):
        B = jnp.einsum("abc, cde -> abde", tt_train[idx], tt_train[idx + 1])
        bonded_tt_train = tt_train[:idx] + [B] + tt_train[idx + 2:]
        grad = self.gradient_functions[idx](*bonded_tt_train, params=params)
        B -= params["lr"] *(grad - jnp.sum(B*grad)*B)
        B_part_1, B_part_2 = part_bond(B)
        tt_train = tt_train[:idx] + [B_part_1, B_part_2] + tt_train[idx + 2:]
        tt_train[idx] = tt_train[idx] / jnp.sqrt(tt_inner_prod(tt_train, tt_train))
        return tt_train

    def _barrier(self, idx):
        equality_constraints = [c(idx) for c in self.equality_constraints]
        inequality_constraints = [c(idx) for c in self.inequality_constraints]

        def penalty(*tt_train, params):
            return params["beta"]/(2*params["mu"] + 1e-3) * sum([jnp.square(c(tt_train)) for c in equality_constraints]) \
                - params["mu"] * sum([jnp.log(c(tt_train) + params["lambda"]) for c in inequality_constraints])

        return penalty

    def _init_tt_train(self):
        tt_train = [2 * np.random.rand(1, 2, 1) - 1 for _ in range(self.dimension)]
        tt_train[0] = tt_train[0] / np.sqrt(tt_inner_prod(tt_train, tt_train))
        tt_train = tt_rank_reduce(tt_train)
        params = {
            "lambda": 1 - self.penalty_function_iq(tt_train),
            "mu": 1.0,
            "beta": 0.0,
            "lr": 1e-2
        }
        return tt_train, params
