from time import time
from utils import *
from optimiser import ILPSolver, AnswerSetSolver

vocab_size = 2
x = Atom(vocab_size, "x")
y = Atom(vocab_size, "y")
h = Hypothesis()
const_space = ConstraintSpace(vocab_size)
e_1 = Boolean_Function(x & y)
const_space.forall_S(h << e_1)
e_1_contradiction = Boolean_Function(~x & y)
const_space.forall_S(h >> e_1_contradiction)
opt = ILPSolver(const_space, vocab_size)
t_1 = time()
hypothesis = opt.find_feasible_hypothesis()
t_2 = time()
print(get_CNF([x, y], hypothesis), flush=True)
print(np.round(tt_to_tensor(hypothesis), decimals=4))
#print("Equality constraint Score: ", [jnp.sum(jnp.abs(c(hypothesis, const_space.expected_truth))) for c in const_space.eq_constraints])
#print("Inequality constraint Score: ", [jnp.sum(c(hypothesis, const_space.expected_truth)) for c in const_space.iq_constraints])
#print("Score:", boolean_criterion(len(hypothesis))(hypothesis), tt_inner_prod(hypothesis, hypothesis))
print(f"Total time taken: {t_2-t_1}s.")
asp_solver = AnswerSetSolver([x, y])
X = asp_solver.get_minimal_answer_set(hypothesis)
print(X)
#print(const_space.noise_op_measure)
#print(const_space.expected_truth)