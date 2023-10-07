import os
import sys

sys.path.append(os.getcwd() + '/../')
from time import time
from src.utils import *
from src.optimiser import ILPSolver

const_space = ConstraintSpace()
number_of_items = 8
number_of_items_log = int(np.log2(number_of_items))
atoms = const_space.generate_atoms(number_of_items_log)
h = const_space.Hypothesis()
target_ranks = [1] + [3 for _ in range(number_of_items_log - 1)] + [1]
value_tensor = tt_normalise(
    [(1 / l_n * 2 * l_np1) * np.random.randn(l_n, 2, l_np1) for l_n, l_np1 in
     zip(target_ranks[:-1], target_ranks[1:])] + [np.array([1, 0]).reshape(1, 2, 1)])
weight = Boolean_Function(const_space, "weight", tt_normalise(
    [(1 / l_n * 2 * l_np1) * np.random.randn(l_n, 2, l_np1) for l_n, l_np1 in
     zip(target_ranks[:-1], target_ranks[1:])] + [np.array([1, 0]).reshape(1, 2, 1)]))
knapsack_capacity = 0.6


def knapsack_objective(tt_train):
    return 1 - tt_inner_prod(value_tensor, tt_train)


const_space.for_all(~weight ^ h, 2 * knapsack_capacity - 1)
opt = ILPSolver(const_space, objective=knapsack_objective)
t_1 = time()
opt.solve()
t_2 = time()
item_tensor = tt_walsh_op(h.value)
items = []
for h in const_space.hypotheses:
    print(f"TT-rank: {h}: {tt_rank(h.value)}")
    print(f"Objective value: {knapsack_objective(h.value)}")
print(f"Total time taken: {t_2 - t_1}s.")
for i in range(number_of_items):
    binar = list(bin(i)[2:])
    binar = [0] * (number_of_items_log - len(binar)) + binar
    con = [np.array([1-int(j), int(j)]).reshape(1, 2, 1) for j in binar]
    if tt_inner_prod(con, item_tensor) > 0:
        items.append(i)
print("Items: ", items)
