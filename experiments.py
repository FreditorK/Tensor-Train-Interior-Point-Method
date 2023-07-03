from time import time

from optimiser import AnswerSetSolver, ILPSolver
from utils import *

vocab_size = 3
x = Atom(vocab_size, "x")
y = Atom(vocab_size, "y")
z = Atom(vocab_size, "z")
func = (x & y & z).to_tt_train()
print(tt_to_tensor(func))
print(tt_shared_influence(func, np.array([0, 1, 2])))