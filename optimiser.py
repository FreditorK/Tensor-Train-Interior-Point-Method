from tt_op import *
from operators import D_func
from utils import Constraint

np.random.seed(7)


class Minimiser:
    def __init__(self, constraints: Constraint, dimension):
        self.dimension = dimension
        self.equality_constraints = constraints._return_forall_constraints()
        self.inequality_constraints = constraints._return_exist_not_constraints() + constraints._return_exists_constraints()
        self.gradient_functions = []
        for idx in range(dimension - 1):
            self.gradient_functions.append(
                partial_D(self._boolean_criterion(idx), idx)
            )
        constraint_functions = [c(-1) for c in self.equality_constraints + self.inequality_constraints]
        self.penalty_function = lambda tt_train: min([c(tt_train) for c in constraint_functions])
        self.complete_gradient = D_func(boolean_criterion(dimension))

    def find_feasible_hypothesis(self):
        tt_train, params = self._init_tt_train()
        criterion = boolean_criterion(self.dimension)
        indices = np.arange(self.dimension - 1)
        prev_max_violation = -np.inf
        constraint_functions = [c(-1) for c in self.equality_constraints + self.inequality_constraints]
        while params["lambda"] > 0:
            for idx in indices:
                tt_train = self._core_iteration(tt_train, params, idx)
                tt_train[idx] = tt_train[idx] / jnp.sqrt(tt_inner_prod(tt_train, tt_train))
            # Projection onto sphere
            # norm_index = np.random.randint(0, self.dimension)
            # tt_train[norm_index] = tt_train[norm_index] / jnp.sqrt(tt_inner_prod(tt_train, tt_train))
            max_violation = self.penalty_function(tt_train)
            print(params["lambda"], [c(tt_train) for c in constraint_functions], tt_inner_prod(tt_train, tt_train))
            params["lambda"] = max(
                params["lambda"] - 0.05, np.abs(min(max_violation, 0))
                # params["lr"]*(3 - 1/(max_violation + params["lambda"]+1e-5)
            )
            # if prev_max_violation > max_violation and max_violation < 0:
            #   params["lr"] *= 0.99
        print(max_violation)
        print("Feasible point found.")
        params["lambda"] = 0
        params["lr"] *= 0.1
        prev_criterion_score = np.inf
        criterion_score = 1.0
        while params["mu"] > 0.0:
            params["mu"] -= 0.005
            for idx in indices:
                tt_train = self._core_iteration(tt_train, params, idx)
            criterion_score = criterion(tt_train)
            print(criterion_score)
            if criterion_score >= prev_criterion_score:
                params["lr"] *= 0.99
            prev_criterion_score = criterion_score
        return tt_train

    def _core_iteration(self, tt_train, params, idx):
        B = jnp.einsum("abc, cde -> abde", tt_train[idx], tt_train[idx + 1])
        bonded_tt_train = tt_train[:idx] + [B] + tt_train[idx + 2:]
        B -= params["lr"] * self.gradient_functions[idx](*bonded_tt_train, params=params)
        B_part_1, B_part_2 = part_bond(B)
        tt_train = tt_train[:idx] + [B_part_1, B_part_2] + tt_train[idx + 2:]
        return tt_train

    def _boolean_criterion(self, idx):
        minus_one = tt_one_bonded(self.dimension, idx)
        minus_one[0] *= -1.0
        penalty = self._barrier(idx)

        def criterion_func(*tt_train, params):
            Ttt_train = tt_bool_op(tt_train)
            squared_Ttt_1 = tt_hadamard(Ttt_train, Ttt_train)
            minus_1_squared_Ttt_1 = tt_add(squared_Ttt_1, minus_one)
            return penalty(tt_train, params) + (1 - params["mu"]) * tt_inner_prod(minus_1_squared_Ttt_1, minus_1_squared_Ttt_1) \

        return criterion_func

    def _barrier(self, idx):
        equality_constraints = [c(idx) for c in self.equality_constraints]
        inequality_constraints = [c(idx) for c in self.inequality_constraints]

        def penalty(tt_train, params):
            return sum([jnp.square(c(tt_train)) for c in equality_constraints]) \
                - params["lambda"] * sum([jnp.log(c(tt_train) + params["lambda"]) for c in inequality_constraints])

        return penalty

    def _init_tt_train(self):
        tt_train = [2 * np.random.rand(1, 2, 1) - 1 for _ in range(self.dimension)]
        tt_train[0] = tt_train[0] / np.sqrt(tt_inner_prod(tt_train, tt_train))
        tt_train = tt_rank_reduce(tt_train)
        params = {
            "lambda": 1 - self.penalty_function(tt_train),
            "mu": 1.0,
            "lr": 1e-2
        }
        return tt_train, params
