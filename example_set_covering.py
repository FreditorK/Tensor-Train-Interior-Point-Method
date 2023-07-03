from functools import reduce
from time import time
from utils import *
from optimiser import ILPSolver, AnswerSetSolver

vocab_size = 10
atoms = np.array(generate_atoms(vocab_size))
set_1 = reduce(lambda x, y: x | y, atoms[[0, 3, 6]])
set_2 = reduce(lambda x, y: x | y, atoms[[1, 2, 6]])
set_3 = reduce(lambda x, y: x | y, atoms[[4, 5]])
set_4 = reduce(lambda x, y: x | y, atoms[[1, 7, 8, 9]])
set_5 = reduce(lambda x, y: x | y, atoms[[0, 2]])
to_cover_set = Boolean_Function(reduce(lambda x, y: x | y, atoms[[0, 2, 3, 4, 5]]))
h = Hypothesis()
const_space = ConstraintSpace(vocab_size)
const_space.forall_S(h >> to_cover_set)

set = set_1 ^ set_2 ^ set_3 ^ set_4 ^ set_5 ^ (set_1 | set_2) ^ (set_2 | set_3) ^ (set_3 | set_4) \
      ^ (set_4 | set_5) ^ (set_1 | set_3) ^ (set_1 | set_4) ^ (set_1 | set_5) ^ (set_2 | set_4) \
      ^ (set_2 | set_5) ^ (set_3 | set_5) ^ (set_1 | set_2 | set_3) ^ (set_1 | set_3 | set_4) \
      ^ (set_3 | set_4 | set_5) ^ (set_1 | set_4 | set_5) ^ (set_1 | set_2 | set_5) \
      ^ (set_1 | set_2 | set_3) ^ (set_2 | set_4 | set_5) ^ (set_2 | set_3 | set_4) \
      ^ (set_1 | set_3 | set_5) ^ (set_1 | set_2 | set_4) ^ (set_1 | set_2 | set_3 | set_5) \
      ^ (set_4 | set_2 | set_3 | set_5) ^ (set_1 | set_4 | set_3 | set_5) ^ (set_1 | set_2 | set_4 | set_5) \
      ^ (set_1 | set_2 | set_3 | set_4 | set_5)
print(tt_to_tensor(tt_bool_op(set.to_tt_train())))