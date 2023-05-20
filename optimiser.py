from tt_op import *
from utils import ConstraintSpace
from utils import NoisyConstraintSpace


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


class ILPSolver:
    def __init__(self, const_space: ConstraintSpace, dimension):
        self.dimension = dimension
        self.const_space = const_space
        self.error_bound = self.const_space.s_lower + 1

    def find_feasible_hypothesis(self):
        error_bound = self.error_bound
        tt_train, params = self._init_tt_train()
        for i in range(100):
            print(f"----------Iteration {i}----------")
            tt_train = self._resolve_constraints(tt_train, params)
            tt_train = self._extract_solution(tt_train)
            criterion_score = self.const_space.eq_crit(tt_train,
                                                       self.const_space.expected_truth) + self.const_space.iq_crit(
                tt_train, self.const_space.expected_truth)
            if criterion_score > error_bound:
                print(f"Constraint Criterion after rounding: {criterion_score}")
                self.const_space.add_faulty_hypothesis(tt_train)
            else:
                return tt_train
        return tt_train

    def _resolve_constraints(self, tt_train, params):
        criterion_score = np.inf
        tt_train = self.const_space.project(tt_train)
        while criterion_score > 2*self.error_bound:  # Gradient induced change, i.e. similar to first-order sufficient condition
            prev_tt_train = tt_train
            tt_train = self._objective(tt_train, params)
            tt_train = self.const_space.project(tt_train)
            criterion_score = self.const_space.radius**2-tt_inner_prod(tt_train, prev_tt_train)
            print(f"Constraint Criterion: {criterion_score} \r", end="")
        print("\n", flush=True)
        return tt_train

    def _extract_solution(self, tt_train):
        criterion_score = self.const_space.boolean_criterion(tt_train)
        while criterion_score > self.error_bound:
            tt_train = self.const_space.round(tt_train)
            criterion_score = self.const_space.boolean_criterion(tt_train)
            print(f"Boolean Criterion: {criterion_score} \r", end="")
        print("\n", flush=True)
        return self.const_space.round(tt_train)

    def _objective(self, tt_train, params):
        for idx in range(self.dimension):
            rank_gradient = self.const_space.rank_gradient(tt_train)[idx]
            tt_train[idx] -= params["lr"] * rank_gradient
            tt_train[idx] += params["lr"] * tt_grad_inner_prod(tt_train, tt_train, rank_gradient, idx) * \
                             tt_train[idx]
            tt_train[idx] *= self.const_space.radius / np.sqrt(tt_inner_prod(tt_train, tt_train))
        return tt_train

    def _init_tt_train(self):
        # Initializes at everything is equivalent formula
        tt_train = [np.random.randn(1, 2, 1) for _ in range(self.dimension)]
        tt_train = tt_mul_scal(1 / jnp.sqrt(tt_inner_prod(tt_train, tt_train)), tt_train)
        params = {
            "lr": 0.08
        }
        return tt_train, params
