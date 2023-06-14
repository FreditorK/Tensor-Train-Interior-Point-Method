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

print("Data Function: ", get_CNF([x, y, z, w, u], e_0.compressed_data), flush=True)
const_space.forall_S(h >> e_0)
opt = ILPSolver(const_space, vocab_size)
t_1 = time()
hypothesis = opt.find_feasible_hypothesis()
t_2 = time()
print("Shapes: ", [t.shape for t in hypothesis])
print(get_CNF([x, y, z, w, u], hypothesis), flush=True)
print("Equality constraint Score: ", [jnp.sum(jnp.abs(c(hypothesis))) for c in const_space.eq_constraints])
print("Inequality constraint Score: ", [jnp.sum(c(hypothesis)) for c in const_space.iq_constraints])
print("Score:", tt_boolean_criterion(len(hypothesis))(hypothesis), tt_inner_prod(hypothesis, hypothesis))
print(f"Total time taken: {t_2-t_1}s.")
asp_solver = AnswerSetSolver([x, y, z, w, u])
X = asp_solver.get_minimal_answer_set(hypothesis)
print(X)
