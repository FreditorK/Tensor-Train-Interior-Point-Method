import numpy as np

from operators import D_func
from utils import *
from tt_op import tt_extract_seq
from optimiser import AnswerSetSolver

vocab_size = 4
x = Atom(vocab_size, "x")
y = Atom(vocab_size, "y")
z = Atom(vocab_size, "z")
w = Atom(vocab_size, "w")
e_tt_1 = (x & (y | z | w)).to_tt_train()
e_tt_2 = ((x & (y | z)) >> w).to_tt_train()
e_tt_3 = (x & (y | z) & w).to_tt_train()
e_tt_4 = (x ^ w).to_tt_train()

print("Zero: ", tt_to_tensor(tt_bool_op(e_tt_1)))
const_space = ConstraintSpace(vocab_size)
tt_rounded = tt_add_noise(e_tt_1)
print("First: ", tt_to_tensor(tt_bool_op(tt_rounded)))
for _ in range(20):
    tt_rounded = const_space.round(tt_rounded)
print("Second: ", tt_to_tensor(tt_bool_op(tt_rounded)))