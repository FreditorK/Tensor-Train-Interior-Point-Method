from time import time
from utils import *
from optimiser import ILPSolver, AnswerSetSolver

const_space = ConstraintSpace()
number_of_items = int(np.log2(8))
atoms = const_space.generate_atoms(number_of_items)  # i.e. graph with 2^3 nodes
h = const_space.Hypothesis()
target_ranks = [1] + [3 for _ in range(number_of_items - 1)] + [1]
value_tensor = tt_normalise(
    [(1 / l_n * 2 * l_np1) * np.random.randn(l_n, 2, l_np1) for l_n, l_np1 in zip(target_ranks[:-1], target_ranks[1:])] + [np.array([1, 0]).reshape(1, 2, 1)])
weight = Boolean_Function(const_space, "weight", tt_normalise(
    [(1 / l_n * 2 * l_np1) * np.random.randn(l_n, 2, l_np1) for l_n, l_np1 in
     zip(target_ranks[:-1], target_ranks[1:])] + [np.array([1, 0]).reshape(1, 2, 1)]))
knapsack_capacity = 0.6
print(const_space.hypotheses)


def knapsack_objective(tt_train):
    return 1 - tt_inner_prod(value_tensor, tt_train)


const_space.for_all(~weight ^ h, 2 * knapsack_capacity - 1)
opt = ILPSolver(const_space, objective=knapsack_objective)
t_1 = time()
opt.solve()
t_2 = time()
for h in const_space.hypotheses:
    print(f"Conjunctive Normal Form: {h}{const_space.atoms} =", h.to_CNF(), flush=True)
    print(f"TT-rank: {h}: {tt_rank(h.value)}")
    print(f"Objective value: {knapsack_objective(h.value)}")
print(f"Total time taken: {t_2 - t_1}s.")
