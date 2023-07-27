from functools import reduce
from time import time

import numpy as np

from utils import *
from optimiser import ILPSolver, AnswerSetSolver

vocab_size = 3
atoms = np.array(generate_atoms(vocab_size))
"""
set_1 = [0, 2, 3, 6]
set_2 = [1, 6]
set_3 = [4]
set_U = [4, 6]
s_1 = reduce(lambda x, y: x | y, atoms[set_1])
s_2 = reduce(lambda x, y: x | y, atoms[set_2])
s_3 = atoms[set_3]
s_u = reduce(lambda x, y: x | y, atoms[set_U])
"""
h_0 = Hypothesis()
h_1 = Hypothesis()
#h_3 = Hypothesis()
expression_train = TensorTrain.from_expression(atoms[0] & ~h_0 & h_1, hypothesis_space=True)
print("Pre-Substitution: ", get_CNF(list(atoms) + [h_0, h_1], expression_train.cores))
h_0_expression = atoms[1] | atoms[2]
h_1_expression = atoms[0]
print(f"To substitute in: h_0: {h_0_expression}, h_1: {h_1_expression}")
h_0.value = TensorTrain.from_expression(h_0_expression).cores
h_1.value = TensorTrain.from_expression(h_1_expression).cores
substituted_expression = h_1.substitute_into(expression_train)
substituted_expression = h_0.substitute_into(substituted_expression)
print("Post-Subsitution:", get_CNF(atoms, substituted_expression.cores))
solution = TensorTrain.from_expression(h_1_expression & ~h_0_expression)
print("Truth", get_CNF(atoms, solution.cores))