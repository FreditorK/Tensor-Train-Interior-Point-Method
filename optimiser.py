from copy import deepcopy

import numpy as np

from tt_op import *
from operators import D_func, partial_D
from utils import ConstraintSpace, Atom


# np.random.seed(7)

class AnswerSetSolver:
    def __init__(self, tt_train: List[np.array], atoms):
        self.tt_train = tt_train
        self.atoms = atoms
        self.dimension = len(tt_train)
        self.params = {
            "lr": 1e-1
        }
        not_ands = ~atoms[0]
        for a in atoms[1:0]:
            not_ands = not_ands & ~a
        self.tt_minimal = not_ands.to_tt_train()

    def _random_sphere_point(self, dimension):
        sphere_x = np.array([np.random.randn() for _ in range(dimension)])
        sphere_x = sphere_x / np.linalg.norm(sphere_x)
        return sphere_x

    def get_answer_set(self, **kwargs):
        set_values = {}
        free_indices = {}
        k = 0
        for i, atom in enumerate(self.atoms):
            if atom.name in kwargs.keys():
                set_values[i] = kwargs[atom.name]
            else:
                free_indices[i] = k
                k += 1
        random_point = self._random_sphere_point(len(free_indices))
        X = [
            np.array([1.0, random_point[free_indices[i]].item()]).reshape(1, 2, 1) if i in free_indices.keys()
            else np.array([1.0, set_values[i]]).reshape(1, 2, 1)
            for i in range(self.dimension)
        ]
        criterion = lambda x: (1 - tt_inner_prod(self.tt_train, x))**2 - tt_inner_prod(self.tt_minimal, x) - 1
        criterion_gradient = D_func(criterion)
        while criterion(X) > 5e-3:
            for _ in range(10):
                for idx in free_indices.keys():
                    X[idx][:, 1] = X[idx][:, 1] - self.params["lr"] * criterion_gradient(X)[idx][:, 1]
            vs = np.array([X[i][:, 1].item() for i in free_indices.keys()])
            vs = vs / np.linalg.norm(vs)
            random_point = self._random_sphere_point(len(free_indices))
            vs = [np.sign(random_point[i] * v) for i, v in enumerate(vs)]
            X = [
                np.array([1.0, vs[free_indices[i]].item()]).reshape(1, 2, 1) if i in free_indices.keys()
                else np.array([1.0, set_values[i]]).reshape(1, 2, 1)
                for i in range(self.dimension)
            ]
        return {self.atoms[i] for i, x in enumerate(X) if int(x[:, 1].item()) == 1}


class Minimiser:
    def __init__(self, const_space: ConstraintSpace, dimension):
        self.dimension = dimension
        self.const_space = const_space
        self.complete_gradient = partial_D(self.criterion(dimension), 0)
        self.weight_tensor = [np.array([1.0, 0.5]).reshape(1, 2, 1) for _ in range(self.dimension)]

    def criterion(self, dimension):
        @jax.jit
        def criterion(*tt_train, params):
            return tt_inner_prod(self.weight_tensor, tt_train)
        return criterion

    def find_feasible_hypothesis(self):
        tt_train, params = self._init_tt_train()
        criterion = boolean_criterion(self.dimension)
        prev_criterion_score = np.inf
        criterion_score = np.inf
        while criterion_score > 1e-4:
            # optimise over first core, we rl_othogonalise in the projections anyway
            tt_train = self.const_space.project(tt_train)
            tt_train = self.const_space.reflect(tt_train)
            tt_train = self._iteration(tt_train, params)
            tt_train = self._round(tt_train, params, iterations=1)
            criterion_score = criterion(tt_train)
            if criterion_score > prev_criterion_score:
                params["lr"] *= 0.99
            prev_criterion_score = criterion_score
            print(f"Current violation: {criterion_score} \r", end="")
        print("\n", flush=True)
        return self._round(tt_train, params)

    def _iteration(self, tt_train, params):
        gradient = self.complete_gradient(*tt_train, params=params)
        tt_train[0] -= params["lr"] * (
            gradient- tt_grad_inner_prod(tt_train, tt_train, gradient, 0) * tt_train[0])
        tt_train[0] = tt_train[0] / jnp.sqrt(tt_inner_prod(tt_train, tt_train))
        return tt_train

    def _round(self, tt_train, params, iterations=10):
        tt_table = tt_bool_op(tt_train)
        for _ in range(iterations):
            tt_table_p3 = tt_hadamard(tt_hadamard(tt_table, tt_table), tt_table)
            tt_table_p3[0] *= params["beta"]
            tt_table[0] *= (1-params["beta"])
            tt_table = tt_rl_orthogonalize(tt_add(tt_table, tt_table_p3))
        tt_train = tt_bool_op_inv(tt_table)
        #tt_train[0] = tt_train[0] / jnp.sqrt(tt_inner_prod(tt_train, tt_train))
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
