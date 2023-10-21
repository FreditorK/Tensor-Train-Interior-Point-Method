from collections import deque
from src.operators import D_func
from src.tt_op import *
from src.utils import ConstraintSpace, stopping_criterion, Hypothesis


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
        self.params = {
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
        for h in self.const_space.hypotheses:
            self.const_space.round(h)
        print("Constraints satisfied: ", self._const_satisfied())

    def _gradient_update(self, hypothesis: Hypothesis):
        h_index = hypothesis.index - self.const_space.atom_count
        target_ranks = tt_ranks(hypothesis.value)
        target_int = np.random.randint(0, len(target_ranks))
        target_ranks[target_int] += 1
        tt_train = tt_add_noise(hypothesis.value, target_ranks=[1] * (self.const_space.atom_count - 1))
        tt_trains_before = [h.value for h in self.const_space.hypotheses[:h_index]]
        tt_trains_after = [h.value for h in self.const_space.hypotheses[h_index + 1:]]
        for idx in range(self.const_space.atom_count):
            gradients = self.objective_grad(tt_trains_before + [tt_train] + tt_trains_after)
            gradient = gradients[h_index][idx]
            tt_train[idx] -= 0.5 * gradient
            tt_train[idx] += tt_grad_inner_prod(tt_train, tt_train, gradient, idx) * tt_train[idx]
            tt_train = tt_normalise(tt_train, idx=idx)
        hypothesis.value = tt_rank_reduce(tt_train)

    def _riemannian_grad(self, hypothesis: Hypothesis):
        #criterions = deque([np.inf], maxlen=self.params["patience"])
        # Gradient induced change, i.e. similar to first-order sufficient condition
        for _ in range(70): # TODO: Need stopping criterion independent from scale, same for rounding, maybe sampling subtensors??
            self._gradient_update(hypothesis)
            self.const_space.project(hypothesis)

    def _round_solution(self, hypothesis: Hypothesis):
        criterion_score = np.inf
        while np.less(self.error_bound, criterion_score):
            self.const_space.round(hypothesis)
            criterion_score = tt_boolean_criterion(hypothesis.value)
