from time import time
from utils import *
from optimiser import ILPSolver, AnswerSetSolver

vocab_size = 2
x = Atom(vocab_size, "x")
y = Atom(vocab_size, "y")
h = Hypothesis()
const_space = NoisyConstraintSpace(vocab_size)
e_1 = Boolean_Function(x & y)
const_space.forall_S(h << e_1)
e_1_contradiction = Boolean_Function(~x & y)
const_space.forall_S(h >> e_1_contradiction)
opt = ILPSolver(const_space, vocab_size)
hypothesis = (x & y).to_tt_train()
p = np.array([1.0, 1.0])
#print("1", tt_to_tensor(hypothesis))
hypothesis = tt_noise_op(hypothesis, p)
print("r", tt_inner_prod(hypothesis, hypothesis))
#hypothesis, _ = opt.const_space.project(hypothesis)
#print("3", tt_to_tensor(hypothesis))
#print("4", tt_to_tensor(tt_noise_op_inv(hypothesis, p)))
print(get_CNF([x, y], tt_noise_op_inv(hypothesis, p)), flush=True)
print("Equality constraint Score: ", [jnp.sum(jnp.abs(c(hypothesis))) for c in const_space.eq_constraints])
print("Inequality constraint Score: ", [jnp.sum(c(hypothesis)) for c in const_space.iq_constraints])
print("Score:", tt_boolean_criterion(len(hypothesis))(tt_noise_op_inv(hypothesis, p)), tt_inner_prod(hypothesis, hypothesis))