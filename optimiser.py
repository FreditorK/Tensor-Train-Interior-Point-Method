from copy import deepcopy

import numpy as np

from tt_op import *
from operators import D_func, partial_D
from utils import ConstraintSpace, Atom


# np.random.seed(7)

class AnswerSetSolver:
    def __init__(self, atoms):
        self.atoms = atoms

    def get_minimal_answer_set(self, tt_train, **assignments):
        set_values = []
        for a in self.atoms:
            if a.name in assignments.keys():
                set_values.append((a.index, assignments[a.name]))
        extracted_seq = tt_extract_seq(tt_train, dict(set_values))
        return {self.atoms[i].name for i, v in enumerate(extracted_seq) if v > 0}


class Minimiser:
    def __init__(self, const_space: ConstraintSpace, dimension):
        self.dimension = dimension
        self.const_space = const_space
        self.soft_loss = lambda *tt_train: sum([loss(tt_train) for loss in self.const_space.iq_constraints])
        self.iq_gradient = partial_D(self.soft_loss, 0)
        self.weight_tensor = [np.array([1.0, 0.5]).reshape(1, 2, 1) for _ in range(self.dimension)]

    def find_feasible_hypothesis(self):
        tt_train, params = self._init_tt_train()
        bool_criterion = boolean_criterion(self.dimension)
        prev_criterion_score = np.inf
        criterion_score = 100
        while np.abs(criterion_score -prev_criterion_score) > 1e-4:
            tt_train, criterion_score = self._iteration(tt_train, params)
            tt_train = self.const_space.project(tt_train)
            if criterion_score > prev_criterion_score:
                params["lr"] *= 0.99
            print(f"Current score: {criterion_score-prev_criterion_score} \r", end="")
            prev_criterion_score = criterion_score
        criterion_score = np.inf
        while criterion_score > 1e-4:
            # optimise over first core, we rl_othogonalise in the projections anyway
            tt_train, _ = self._iteration(tt_train, params)
            tt_train = self._round(tt_train, params)
            tt_train = self.const_space.project(tt_train)
            criterion_score = bool_criterion(tt_train)
            print(f"Current violation: {criterion_score} \r", end="")
        print("\n", flush=True)
        return self._round(tt_train, params)

    def _iteration(self, tt_train, params):
        gradient = self.iq_gradient(*tt_train)
        tt_train[0] -= params["lr"] * (
            gradient- tt_grad_inner_prod(tt_train, tt_train, gradient, 0) * tt_train[0])
        tt_train[0] = tt_train[0] / jnp.sqrt(tt_inner_prod(tt_train, tt_train))
        return tt_train, jnp.sum(jnp.square(gradient))

    def _round(self, tt_train, params):
        tt_table = tt_bool_op(tt_train)
        tt_table_p3 = tt_hadamard(tt_hadamard(tt_table, tt_table), tt_table)
        tt_table_p3[0] *= params["beta"]
        tt_table[0] *= -params["beta"]
        tt_table = tt_rl_orthogonalize(tt_add(tt_table, tt_table_p3))
        tt_update = tt_bool_op_inv(tt_table)
        tt_train[0] *= (1 - tt_inner_prod(tt_update, tt_train))
        tt_train = tt_add(tt_update, tt_train)
        tt_train[0] = tt_train[0] / jnp.sqrt(tt_inner_prod(tt_train, tt_train))
        return tt_rl_orthogonalize(tt_train)

    def _init_tt_train(self):
        # Initializes at everything is equivalent formula
        tt_train = [np.array([0.5, 1.0]).reshape(1, 2, 1) for _ in range(self.dimension)]
        tt_train[0] = tt_train[0] / jnp.sqrt(tt_inner_prod(tt_train, tt_train))
        params = {
            "lr": 5e-3,
            "beta": -0.5
        }
        return tt_train, params
