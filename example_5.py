from time import time
from utils import *
from optimiser import ILPSolver, AnswerSetSolver

vocab_size = 5
x = Atom(vocab_size, "x")
y = Atom(vocab_size, "y")
z = Atom(vocab_size, "z")
w = Atom(vocab_size, "w")
u = Atom(vocab_size, "u")
h = Hypothesis()
const_space = ConstraintSpace(vocab_size)
e_0 = Boolean_Data(
    np.array([
        [1, 0, 0, 1, 1],
        [1, 0, 1, 0, 1],
        [1, 0, 1, 0, 1],
        [1, 0, 1, 1, 1],
        [1, 0, 0, 0, 1],
        [1, 0, 0, 1, 1],
        [1, 0, 1, 1, 1],
        [1, 0, 0, 1, 0]
    ]),
    np.array([1, 0, 0, 0, 1, 1, 0, 1])
)

print(get_CNF([x, y, z, w, u], e_0.to_tt_train()), flush=True)
asp_solver = AnswerSetSolver([x, y, z, w, u])
X = asp_solver.get_minimal_answer_set(e_0.to_tt_train())
print(X)