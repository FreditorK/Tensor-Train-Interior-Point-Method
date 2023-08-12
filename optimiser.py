from collections import deque

import numpy as np
from operators import D_func
from tt_op import *
from utils import ConstraintSpace, TTExpression
from functools import partial


# np.random.seed(7)

class AnswerSetSolver:
    def __init__(self, const_space: ConstraintSpace):
        self.atoms = const_space.atoms

    def get_minimal_answer_set(self, tt_train, **assignments):
        set_values = []
        for a in self.atoms:
            if a.name in assignments.keys():
                set_values.append((a.index, assignments[a.name]))
        extracted_seq = tt_extract_seq(tt_train, dict(set_values))
        return {self.atoms[i].name for i, v in enumerate(extracted_seq) if v > 0}


class ILPSolver:
    def __init__(self, const_space: ConstraintSpace, objective=None):
        self.objective_grad = None
        if objective is not None:
            self.objective_grad = D_func(lambda tt_trains: objective(*tt_trains))
        self.const_space = const_space
        self.error_bound = 2 ** (-self.const_space.atom_count)
        self.boolean_criterion = tt_boolean_criterion(self.const_space.atom_count)
        self.params = {
            "orig_lr": 0.075,
            "lr": 0.075,
            "noise": self.error_bound/3,
            "patience": 5
        }

    def _const_satisfied(self):
        is_satisfied = True
        for h in self.const_space.hypotheses:
            for c in self.const_space.eq_constraints[h]:
                is_satisfied = is_satisfied & c.is_satisfied(h.value)
            for c in self.const_space.iq_constraints[h]:
                is_satisfied = is_satisfied & c.is_satisfied(h.value)
        return is_satisfied

    def solve(self):
        iter_function = self._project if self.objective_grad is None else self._riemannian_grad
        print("Solving relaxation...")
        iter_function()
        print("Rounding solution...")
        self._round_solution()
        while not self._const_satisfied():
            for h in self.const_space.hypotheses:
                print(h, h.to_CNF())
                self.const_space.add_exclusion(h)
            self.params["lr"] = self.params["orig_lr"]
            print("Solving relaxation...")
            iter_function()
            print("Rounding solution...")
            self._round_solution()

    def _gradient_update(self):
        tt_trains = [tt_add_noise(h.value, rank=1, noise_radius=self.params["noise"]) for h in
                     self.const_space.hypotheses]
        for idx in range(self.const_space.atom_count):
            gradients = self.objective_grad(tt_trains)
            for i in range(self.const_space.hypothesis_count):
                gradient = gradients[i][idx]
                tt_trains[i][idx] -= self.params["lr"] * gradient
                tt_trains[i][idx] += self.params["lr"] * tt_grad_inner_prod(tt_trains[i], tt_trains[i], gradient, idx) * \
                                     tt_trains[i][idx]
                tt_trains[i] = tt_normalise(tt_trains[i], idx=idx)
        for h, tt_train in zip(self.const_space.hypotheses, tt_trains):
            h.value = tt_rank_reduce(tt_train)

    def _project(self):
        for h in self.const_space.permuted_hypotheses:
            self.const_space.project(h)

    def _riemannian_grad(self):
        criterions = deque([np.inf], maxlen=self.params["patience"])
        # Gradient induced change, i.e. similar to first-order sufficient condition
        while np.array(criterions)[-3:].mean() >= (self.params["lr"]/8)*self.error_bound:
            hypotheses_copies = deepcopy([h.value for h in self.const_space.hypotheses])
            self._gradient_update()
            self._project()
            criterion = self.const_space.stopping_criterion([h.value for h in self.const_space.hypotheses],
                                                            hypotheses_copies)
            criterions.append(criterion)
            if (np.array(criterions)[:-1] - np.array(criterions)[1:]).mean() > 0:
                self.params["lr"] *= 0.99

    def _round_solution(self):
        criterion_score = np.mean([self.boolean_criterion(h.value) for h in self.const_space.hypotheses])
        while criterion_score > self.error_bound:
            for h in self.const_space.hypotheses:
                self.const_space.round(h, 2*self.error_bound)
                h.value = tt_rank_retraction([core.shape[-1] for core in h.value[:-1]], h.value)
            criterion_score = np.mean([self.boolean_criterion(h.value) for h in self.const_space.hypotheses])
        for h in self.const_space.hypotheses:
            self.const_space.round(h, 0)
            h.value = tt_rank_reduce(h.value)
