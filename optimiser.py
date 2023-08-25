from collections import deque

import numpy as np
import scipy
from matplotlib import pyplot as plt

from operators import D_func
from tt_op import *
from utils import ConstraintSpace, TTExpression, stopping_criterion, Hypothesis
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
            "orig_lr": 1.9 * self.error_bound,
            "lr": 1.9 * self.error_bound,
            # It cannot be lower, otherwise it skips functions, i.e. distance between functions is error_bound
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
        iter_function = self.const_space.project if self.objective_grad is None else self._riemannian_grad
        for hypothesis in self.const_space.permuted_hypotheses:
            iter_function(hypothesis)
            self._round_solution(hypothesis)
        while not self._const_satisfied():
            print([(str(h), h.to_CNF()) for h in self.const_space.hypotheses])
            self.const_space.extend_repeller()
            for hypothesis in self.const_space.permuted_hypotheses:
                iter_function(hypothesis)
                self._round_solution(hypothesis)
        for h in self.const_space.hypotheses:
            self.const_space.round(h, 0)

    def _gradient_update(self, hypothesis: Hypothesis):
        h_index = hypothesis.index - self.const_space.atom_count
        tt_train = tt_add_noise(hypothesis.value, self.error_bound / 3, target_ranks=[1] * (self.const_space.atom_count - 1))
        tt_trains_before = [h.value for h in self.const_space.hypotheses[:h_index]]
        tt_trains_after = [h.value for h in self.const_space.hypotheses[h_index + 1:]]
        for idx in range(self.const_space.atom_count):
            gradients = self.objective_grad(tt_trains_before + [tt_train] + tt_trains_after)
            gradient = gradients[h_index][idx]
            tt_train[idx] -= self.params["lr"] * gradient
            tt_train[idx] += self.params["lr"] * tt_grad_inner_prod(tt_train, tt_train, gradient, idx) * tt_train[idx]
            tt_train = tt_normalise(tt_train, idx=idx)
        hypothesis.value = tt_rank_reduce(tt_train)

    def _riemannian_grad(self, hypothesis: Hypothesis):
        criterions = deque([np.inf], maxlen=self.params["patience"])
        # Gradient induced change, i.e. similar to first-order sufficient condition
        while np.array(criterions)[-3:].mean() >= (self.params["lr"] / 6) * self.error_bound:
            hypothesis_copy = deepcopy(hypothesis.value)
            self._gradient_update(hypothesis)
            self.const_space.project(hypothesis)
            criterions.append(stopping_criterion(hypothesis.value, hypothesis_copy))
            if (np.array(criterions)[:-1] - np.array(criterions)[1:]).mean() > 0:
                self.params["lr"] = max(0.99 * self.params["lr"], 0.5 * self.error_bound)
        self.params["lr"] = self.params["orig_lr"]

    def _round_solution(self, hypothesis: Hypothesis):
        criterion_score = np.inf
        while criterion_score >= self.error_bound:
            self.const_space.round(hypothesis, self.error_bound)
            criterion_score = self.boolean_criterion(hypothesis.value)
            print(f"Boolean Criterion: {criterion_score} \r", end="")
        print("\n", flush=True)
