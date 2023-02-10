from tt_op import *
from operators import D_func

#np.random.seed(7)


class Minimiser:
    def __init__(self, constraints, dimension):
        self.dimension = dimension
        self.constraints = constraints
        self.gradient_functions = []
        for idx in range(dimension - 1):
            self.gradient_functions.append(
                partial_D(self._boolean_criterion(idx), idx)
            )
        constraint_functions = [c(-1) for c in self.constraints]
        self.penalty_function = lambda tt_train: min([c(tt_train) for c in constraint_functions]+ [1 - tt_inner_prod(tt_train, tt_train)])
        self.complete_gradient = D_func(boolean_criterion(dimension))

    def find_feasible_hypothesis(self):
        tt_train, params = self._init_tt_train()
        criterion = boolean_criterion(self.dimension)
        indices = np.arange(self.dimension - 1)
        prev_max_violation = -np.inf
        constraint_functions = [c(-1) for c in self.constraints]
        while params["lambda"] > 0:
            for idx in indices:
                tt_train = self._core_iteration(tt_train, params, idx)
            max_violation = self.penalty_function(tt_train)
            print(params["lambda"], [c(tt_train) for c in constraint_functions]+ [1 - tt_inner_prod(tt_train, tt_train)])
            params["lambda"] = max(
                params["lambda"] - params["lr"]*(2 - params["mu"]*params["lambda"]/(max_violation + params["lambda"]) - params["mu"]*jnp.log(max_violation + params["lambda"])),
                np.abs(max_violation)+1e-3
            )
            if prev_max_violation > max_violation and max_violation < 0:
                params["lr"] *= 0.99
        print("Feasible point found.")
        params["lambda"] = 0
        params["mu"] = 0.5
        params["lr"] *= 0.1
        prev_criterion_score = np.inf
        criterion_score = 1.0
        while criterion_score > 1e-5:
            gradient = self.complete_gradient(tt_train)
            tt_train = [t - params["lr"] * gradient[i] for i, t in enumerate(tt_train)]
            criterion_score = criterion(tt_train)
            if criterion_score >= prev_criterion_score:
                params["lr"] *= 0.99
            prev_criterion_score = criterion_score

        return tt_rank_reduce(tt_train)

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
            Ttt_train = tt_bool_op(tt_train)  # TODO: Ttt_train is tt_train on git
            squared_Ttt_1 = tt_hadamard(Ttt_train, Ttt_train)
            minus_1_squared_Ttt_1 = tt_add(squared_Ttt_1, minus_one)
            return (1 - params["mu"]) * tt_inner_prod(minus_1_squared_Ttt_1, minus_1_squared_Ttt_1) \
                - params['mu'] * params['lambda']*penalty(tt_train, params['lambda']) \
                - (params["mu"] - 0.5) * params['lambda'] * jnp.log(1 - tt_inner_prod(tt_train, tt_train) + params["lambda"])

        return criterion_func

    def _barrier(self, idx):
        constraints = [c(idx) for c in self.constraints]

        def penalty(tt_train, shift):
            return sum([jnp.log(c(tt_train) + shift) for c in constraints])

        return penalty

    def _init_tt_train(self):
        tt_train = [2 * np.random.rand(1, 2, 1) - 1 for _ in range(self.dimension)]
        params = {
            "lambda": 1 - self.penalty_function(tt_train),
            "mu": 1.0,
            "lr": 1e-1
        }
        return tt_train, params
