import numpy as np
from operators import D_func
from tt_op import *
from utils import ConstraintSpace
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
            "lr": 0.075,
            "noise": 0.9 * self.error_bound
        }

    def _const_satisfied(self):
        is_satisfied = True
        for h in self.const_space.hypotheses:
            for c in self.const_space.eq_constraints[h]:
                is_satisfied = is_satisfied & c.is_satisfied(h.value)
            for c in self.const_space.iq_constraints[h]:
                is_satisfied = is_satisfied & c.is_satisfied(h.value)
        return is_satisfied

    def solve(self, timeout=100):
        print("Optimising the relaxation...")
        if self.objective_grad is None:
            self._project()
        else:
            self._riemannian_grad()
        print("Rounding the solution...")
        self._round_solution()
        if not self._const_satisfied():
            self._stir_up(timeout)

    def _stir_up(self, timeout):
        for i in range(timeout):
            print(f"----------Stir up {i}----------")
            self._project()
            for h in self.const_space.hypotheses:
                rank = tt_rank(h.value)
                h.value = tt_add_noise(h.value, rank=rank, noise_radius=0.5)
            self._round_solution()
            if self._const_satisfied():
                break

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
            h.value = tt_train

    def _project(self):
        for h in self.const_space.permuted_hypotheses:
            self.const_space.project(h)

    def _init_hypotheses(self):
        criterion = np.inf
        while criterion >= 0.1 * self.error_bound * self.params["lr"]:
            hypotheses_copies = deepcopy([h.value for h in self.const_space.hypotheses])
            self._gradient_update()
            criterion = self.const_space.stopping_criterion([h.value for h in self.const_space.hypotheses],
                                                            hypotheses_copies)
            print(f"Stopping Criterion: {criterion} \r", end="")
        print("\n", flush=True)

    def _riemannian_grad(self):
        criterion = np.inf
        self._init_hypotheses()
        while criterion >= 0.1 * self.error_bound * self.params["lr"]:  # Gradient induced change, i.e. similar to first-order sufficient condition
            hypotheses_copies = deepcopy([h.value for h in self.const_space.hypotheses])
            self._gradient_update()
            self._project()
            criterion = self.const_space.stopping_criterion([h.value for h in self.const_space.hypotheses],
                                                            hypotheses_copies)
            print(f"Stopping Criterion: {criterion} \r", end="")
        print("\n", flush=True)

    def _round_solution(self):
        criterion_score = np.mean([self.boolean_criterion(h.value) for h in self.const_space.hypotheses])
        while criterion_score > self.error_bound:
            for h in self.const_space.hypotheses:
                self.const_space.round(h)
                h.value = tt_rank_retraction([core.shape[-1] for core in h.value[:-1]], h.value)
            criterion_score = np.mean([self.boolean_criterion(h.value) for h in self.const_space.hypotheses])
            print(f"Boolean Criterion: {criterion_score} \r", end="")
        print("\n", flush=True)
        for h in self.const_space.hypotheses:
            self.const_space.round(h)
            h.value = tt_rank_reduce(h.value)
