from functools import reduce
from time import time

import numpy as np

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
