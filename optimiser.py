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
            "lr": 0.08,
            "noise": 0.5*self.error_bound
        }

    def solve(self, timeout=100):
        iter_function = self.const_space.project if self.objective_grad is None else self._riemannian_grad
        for i in range(timeout):
            print(f"----------Iteration {i}----------")
            iter_function()
            self._round_solution()
            criterion_score = self.const_space.stopping_criterion()
            if criterion_score < self.error_bound:
                break
            print(f"Constraint Criterion after rounding: {criterion_score}")
            self._stir_up()

    def _stir_up(self):
        for h in self.const_space.hypotheses:
            rank = tt_rank(h.value)
            h.value = tt_add_noise(h.value, rank=rank, noise_radius=self.error_bound)

    def _gradient_update(self):
        tt_trains = [tt_add_noise(h.value, rank=1, noise_radius=self.params["noise"]) for h in self.const_space.hypotheses]
        for idx in range(self.const_space.atom_count):
            gradients = self.objective_grad(tt_trains)
            for i in range(self.const_space.hypothesis_count):
                gradient = gradients[i][idx]
                tt_trains[i][idx] -= self.params["lr"] * gradient
                tt_trains[i][idx] += self.params["lr"] * tt_grad_inner_prod(tt_trains[i], tt_trains[i], gradient, idx) * \
                             tt_trains[i][idx]
                tt_trains[i] = self.const_space.normalise(tt_trains[i], idx)
        for h, tt_train in zip(self.const_space.hypotheses, tt_trains):
            h.value = tt_train

    def _init_hypotheses(self):
        criterion = np.inf
        while criterion >= 0.1*self.error_bound * self.params["lr"]:
            hypotheses_copies = deepcopy([h.value for h in self.const_space.hypotheses])
            self._gradient_update()
            criterion = self.const_space.stopping_criterion(self.const_space.hypotheses, hypotheses_copies)
            print(f"Stopping Criterion: {criterion} \r", end="")
        print("\n", flush=True)

    def _riemannian_grad(self):
        criterion = np.inf
        self._init_hypotheses()
        while criterion >= 0.1*self.error_bound*self.params["lr"]:  # Gradient induced change, i.e. similar to first-order sufficient condition
            hypotheses_copies = deepcopy([h.value for h in self.const_space.hypotheses])
            self._gradient_update()
            self.const_space.project()
            criterion = self.const_space.stopping_criterion(self.const_space.hypotheses, hypotheses_copies)
            print(f"Stopping Criterion: {criterion} \r", end="")
        print("\n", flush=True)

    def _round_solution(self):
        criterion_score = np.mean([self.boolean_criterion(h.value) for h in self.const_space.hypotheses])
        retraction = partial(tt_rank_retraction, )
        while criterion_score > self.error_bound:
            for h in self.const_space.hypotheses:
                self.const_space.round(h)
                h.value = retraction([core.shape[-1] for core in h.value[:-1]], h.value)
            criterion_score = np.mean([self.boolean_criterion(h.value) for h in self.const_space.hypotheses])
            print(f"Boolean Criterion: {criterion_score} \r", end="")
        print("\n", flush=True)
        for h in self.const_space.hypotheses:
            self.const_space.round(h)
            h.value = tt_rank_reduce(h.value)
