from functools import reduce
from time import time
from utils import *
from optimiser import ILPSolver, AnswerSetSolver

vocab_size = 7
atoms = np.array(generate_atoms(vocab_size))
set_1 = [0, 2, 3, 6]
set_2 = [1, 6]
set_3 = [4]
set_U = [4, 6]
s_1 = reduce(lambda x, y: x | y, atoms[set_1])
s_2 = reduce(lambda x, y: x | y, atoms[set_2])
s_3 = atoms[set_3]
s_u = reduce(lambda x, y: x | y, atoms[set_U])
h_1 = Hypothesis(indices=set_1)
h_2 = Hypothesis(indices=set_2)
h_3 = Hypothesis(indices=set_3)
h_1.active = True
h_2.active = False
h_3.active = False
a = (h_1 & h_2 & s_1).to_tt_train()
print(a)
print(b)
#const_space = ConstraintSpace(vocab_size)