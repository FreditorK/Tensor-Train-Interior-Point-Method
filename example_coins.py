from time import time
from utils import *
from optimiser import ILPSolver, AnswerSetSolver

const_space = ConstraintSpace()
head_c1 = const_space.Atom("head(c_1)")
tail_c1 = const_space.Atom("tail(c_1)")
tail_c2 = const_space.Atom("tail(c_2)")
head_c2 = const_space.Atom("head(c_2)")
h = const_space.Hypothesis("h_0")


def symmetry_objective(tt_train):
    return -tt_inner_prod(tt_train, [jnp.swapaxes(t, 0, -1) for t in tt_train[::-1]])


#print(symmetry_objective(TTExpression.from_expression((head_c1 ^ tail_c1) & (head_c2 ^ tail_c2)).cores))
#print(symmetry_objective(TTExpression.from_expression((head_c1 ^ tail_c1)).cores))


const_space.for_all(h >> (head_c1 ^ tail_c1))
const_space.there_exists(~(h << (head_c1 ^ tail_c1)))
opt = ILPSolver(const_space, objective=symmetry_objective)
t_1 = time()
hypothesis = opt.solve()
t_2 = time()
print("TT-rank: ", [f"{h}: {tt_rank(h.value)}" for h in const_space.hypotheses])
print(f"Conjunctive Normal Form: h{const_space.atoms} =", h.to_CNF(), flush=True)
print(f"Total time taken: {t_2 - t_1}s.")
