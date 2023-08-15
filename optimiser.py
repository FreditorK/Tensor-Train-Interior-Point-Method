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
            "orig_lr": 1.9*self.error_bound,
            "lr": 1.9*self.error_bound, # It cannot be lower, otherwise it skips functions, i.e. distance between functions is error_bound
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
            print([(str(h), h.to_CNF()) for h in self.const_space.hypotheses])
            self.const_space.extend_repeller()
            self.params["lr"] = self.params["orig_lr"]
            print("Solving relaxation...")
            iter_function()
            print("Rounding solution...")
            self._round_solution()
        for h in self.const_space.hypotheses:
            self.const_space.round(h, 0)

    def _gradient_update(self):
        h = self.const_space.random_hypothesis()
        h_index = h.index-self.const_space.atom_count
        tt_train = tt_add_noise(h.value, self.error_bound/3, rank=1)
        tt_trains = [h.value for h in self.const_space.hypotheses[:h_index]]
        for idx in range(self.const_space.atom_count):
            gradients = self.objective_grad(tt_trains[:h_index] + [tt_train] + tt_trains[h_index+1:])
            gradient = gradients[h_index][idx]
            tt_train[idx] -= self.params["lr"] * gradient
            tt_train[idx] += self.params["lr"] * tt_grad_inner_prod(tt_train, tt_train, gradient, idx) * tt_train[idx]
            tt_train = tt_normalise(tt_train, idx=idx)
        h.value = tt_rank_reduce(tt_train)

    def _project(self):
        h = self.const_space.random_hypothesis()
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
                self.params["lr"] = max(0.99*self.params["lr"], 0.5*self.error_bound)

    def _round_solution(self):
        criterion_score = np.mean([self.boolean_criterion(h.value) for h in self.const_space.hypotheses])
        while criterion_score > self.error_bound:
            for h in self.const_space.hypotheses:
                self.const_space.round(h, self.error_bound)
                h.value = tt_rank_retraction([core.shape[-1] for core in h.value[:-1]], h.value)
            criterion_score = np.mean([self.boolean_criterion(h.value) for h in self.const_space.hypotheses])
