from time import time
from utils import *
from optimiser import Minimiser, AnswerSetSolver

vocab_size = 3
x = Atom(vocab_size, "x")
y = Atom(vocab_size, "y")
z = Atom(vocab_size, "z")
h = Hypothesis()
const_space = ConstraintSpace()
e_1 = Boolean_Function(x & y)
const_space.forall_S(h << e_1)
e_1_contradiction = Boolean_Function(~x & y)
const_space.forall_S(h >> e_1_contradiction)
top = tt_leading_one(3)
top_p = tt_leading_one(3)
k = 0.72
top_p[0] *= k
bot = tt_leading_one(3)
bot_p = tt_leading_one(3)
bot[0] *= -1.0
bot_p[0] *= -k
a = (x & y).to_tt_train()
ps = np.array([0.0, 1.0, 1.0])
a = tt_noise_op(a, ps)
#e_1 = tt_noise_op(e_1, ps)
#e_1_contradiction = tt_noise_op(e_1_contradiction, ps)
#print(const_space.eq_constraints[1](const_space.projections[0](a)))
print(tt_inner_prod(tt_add(e_1.tt_example, top_p), tt_add(a, bot_p)))
print(tt_inner_prod(tt_add(e_1_contradiction.tt_example, bot_p), tt_add(a, top_p)))
#print(const_space.eq_constraints[0](const_space.projections[1](a)))
"""
opt = Minimiser(const_space, vocab_size)
t_1 = time()
hypothesis = opt.find_feasible_hypothesis()
t_2 = time()
print(get_CNF([x, y], hypothesis), flush=True)
print(np.round(tt_to_tensor(hypothesis), decimals=4))
print("Equality constraint Score: ", [jnp.sum(jnp.abs(c(hypothesis))) for c in const_space.eq_constraints])
print("Inequality constraint Score: ", [jnp.sum(c(hypothesis)) for c in const_space.iq_constraints])
print("Score:", boolean_criterion(len(hypothesis))(hypothesis), tt_inner_prod(hypothesis, hypothesis))
print(f"Total time taken: {t_2-t_1}s.")
asp_solver = AnswerSetSolver([x, y])
X = asp_solver.get_minimal_answer_set(hypothesis)
print(X)
print(const_space.noise_op_measure)
"""