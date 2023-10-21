import os
import sys

sys.path.append(os.getcwd() + '/../')
from time import time
from src.utils import *
from src.optimiser import ILPSolver
from src.tt_data_op import *

file_name = "../data/knapsack.csv"
tt_value, num_items = data_to_tt(file_name, 0)  # TODO: Does not return Boolean function, look into data_to_tt
tt_weight, num_items_2 = data_to_tt(file_name, 1)
knapsack_capacity = 0.6
assert num_items_2 == num_items, "Faulty data!"
const_space = ConstraintSpace()
atoms = const_space.generate_atoms(len(tt_value))
h = const_space.Hypothesis()
weight_function = Boolean_Function(const_space, "weight", tt_weight + [np.array([1, 0]).reshape(1, 2, 1)])


def knapsack_objective(tt_train):
    return 1 - tt_inner_prod(tt_value, tt_train)


const_space.for_all(~weight_function ^ h, 2 * knapsack_capacity - 1)
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
number_of_items_log = int(np.log2(num_items))
for i in range(num_items):
    binar = list(bin(i)[2:])
    binar = [0] * (number_of_items_log - len(binar)) + binar
    con = [np.array([1 - int(j), int(j)]).reshape(1, 2, 1) for j in binar]
    if tt_inner_prod(con, item_tensor) > 0:
        items.append(i)
print("Items: ", items)
