from copy import deepcopy

import numpy as np

from tt_op import *
from operators import D_func
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
        self.complete_gradient = D_func(self.criterion(dimension))
        # self.weight_tensor = [np.array([0.5, 1.0]).reshape(1, 2, 1) for _ in range(self.dimension)]

    def criterion(self, dimension):
        one = tt_one(dimension)
        one[0] *= -1.0

        @jax.jit
        def criterion(tt_train, params):
            tt_train = tt_bool_op(tt_train)
            squared_Ttt_1 = tt_hadamard(tt_train, tt_train)
            minus_1_squared_Ttt_1 = tt_add(squared_Ttt_1, one)
            return tt_inner_prod(minus_1_squared_Ttt_1, minus_1_squared_Ttt_1) \
                # - params["mu"]*tt_inner_prod(self.weight_tensor, tt_train)

        return criterion

    def find_feasible_hypothesis(self):
        tt_train, params = self._init_tt_train()
        criterion = boolean_criterion(self.dimension)
        prev_criterion_score = np.inf
        criterion_score = np.inf
        while criterion_score > 1e-4:
            tt_train = self.const_space.project(tt_train)
            tt_train = self.const_space.reflect(tt_train)
            for idx in range(self.dimension):
                tt_train = self._iteration(tt_train, params, idx)
            criterion_score = criterion(tt_train)
            if criterion_score > prev_criterion_score:
                params["lr"] *= 0.99
            prev_criterion_score = criterion_score
            # params["mu"] = max(params["mu"] - 0.01, 0)
            print(f"Current violation: {criterion_score} \r", end="")
        print("\n", flush=True)
        return tt_train

    """
    def _bonded_iteration(self, tt_train, params, idx):
        B = jnp.einsum("abc, cde -> abde", tt_train[idx], tt_train[idx + 1])
        bonded_tt_train = tt_train[:idx] + [B] + tt_train[idx + 2:]
        grad = self.gradient_functions[idx](*bonded_tt_train, params=params)
        B -= params["lr"] *(grad - tt_grad_inner_prod(bonded_tt_train, bonded_tt_train, grad, idx)*B)
        B_part_1, B_part_2 = part_bond(B)
        tt_train = tt_train[:idx] + [B_part_1, B_part_2] + tt_train[idx + 2:]
        tt_train[idx] = tt_train[idx] / jnp.sqrt(tt_inner_prod(tt_train, tt_train))
        return tt_train
    """

    def _iteration(self, tt_train, params, idx):
        gradient = self.complete_gradient(tt_train, params)[idx]
        tt_train[idx] -= params["lr"] * (
            gradient - tt_grad_inner_prod(tt_train, tt_train, gradient, idx) * tt_train[idx])
        tt_train[idx] = tt_train[idx] / jnp.sqrt(tt_inner_prod(tt_train, tt_train))
        return tt_train

    def _init_tt_train(self):
        # Initializes at everything is equivalent formula
        tt_train = [np.array([1 / np.sqrt(5), 2 / np.sqrt(5)]).reshape(1, 2, 1) for _ in
                    range(self.dimension)]  # [2 * np.random.rand(1, 2, 1) - 1 for _ in range(self.dimension)]
        # tt_train[0] = tt_train[0] / np.sqrt(tt_inner_prod(tt_train, tt_train))
        """
        print("Before", [c(tt_train) for c in self.const_space.eq_constraints])
        tt_train = self.const_space.project(tt_train)
        print("After", [c(tt_train) for c in self.const_space.eq_constraints])
        tt_train[0]*=1.5
        print("Before", [c(tt_train) for c in self.const_space.eq_constraints])
        tt_train = self.const_space.project(tt_train)
        print("After", [c(tt_train) for c in self.const_space.eq_constraints])
        """
        params = {
            "lr": 3e-3
        }
        return tt_train, params
