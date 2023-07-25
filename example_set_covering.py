from functools import reduce
from time import time
from utils import *
from optimiser import ILPSolver, AnswerSetSolver

vocab_size = 6
atoms = np.array(generate_atoms(vocab_size))
set_1 = [0, 2, 3, 6]
set_2 = [1, 6]
set_3 = [4]
set_U = [4, 6]
s_1 = reduce(lambda x, y: x | y, atoms[set_1])
s_2 = reduce(lambda x, y: x | y, atoms[set_2])
s_3 = atoms[set_3]
to_cover_set = Boolean_Function(reduce(lambda x, y: x | y, atoms[set_U]))
h_1 = Hypothesis(len(set_1))
h_2 = Hypothesis(len(set_2))
h_3 = Hypothesis(len(set_3))
const_space = ConstraintSpace(vocab_size)