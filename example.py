from time import time
from utils import *
from optimiser import Minimiser

vocab_size = 3
x = Atom(3, "x")
y = Atom(3, "y")
z = Atom(3, "z")
h = Hypothesis()
e_1 = Boolean_Function(x & y & z)
const_space = ConstraintSpace()
const_space.forall_S(h << e_1)
#e_2 = Boolean_Function(x << (y & z))
#const_space.exists_S(h << e_2)
opt = Minimiser(const_space, vocab_size)
t_1 = time()
hypothesis = opt.find_feasible_hypothesis()
t_2 = time()
print(get_CNF([x, y, z], hypothesis), flush=True)
print(np.round(tt_to_tensor(hypothesis), decimals=5))
print("Equality constraint Score: ", [jnp.sum(jnp.abs(c(hypothesis))) for c in const_space.eq_constraints])
print("Inequality constraint Score: ", [jnp.sum(c(-1)(hypothesis)) for c in opt.inequality_constraints])
print("Score:", boolean_criterion(len(hypothesis))(hypothesis), tt_inner_prod(hypothesis, hypothesis))
print(f"Total time taken: {t_2-t_1}s.")