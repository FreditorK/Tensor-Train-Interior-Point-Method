from time import time
from utils import *
from optimiser import Minimiser

vocab_size = 4
head_coin_1 = Atom(vocab_size, "head_coin_1")
tail_coin_1 = Atom(vocab_size, "tail_coin_1")
head_coin_2 = Atom(vocab_size, "head_coin_2")
tail_coin_2 = Atom(vocab_size, "tail_coin_2")
h = Hypothesis()
e_1 = Boolean_Function(
    ((head_coin_1 & ~tail_coin_1) | (~head_coin_1 & tail_coin_1)) &
    ((head_coin_2 & ~tail_coin_2) | (~head_coin_2 & tail_coin_2))
)
e_2 = Boolean_Function(
    ~(
    (head_coin_1 & tail_coin_1) | (~head_coin_1 & ~tail_coin_1) &
    (head_coin_2 & tail_coin_2) | (~head_coin_2 & ~tail_coin_2)
    )
)
const_space = ConstraintSpace()
const_space.forall_S(h >> e_1)  # True e_1 implies true h
const_space.forall_S(h >> e_2)  # True e_2 implies false h
opt = Minimiser(const_space, vocab_size)
t_1 = time()
hypothesis = opt.find_feasible_hypothesis()
t_2 = time()
print("Equality constraint Score: ", [jnp.sum(jnp.abs(c(hypothesis))) for c in const_space.eq_constraints])
print("Inequality constraint Score: ", [jnp.sum(c(hypothesis)) for c in const_space.iq_constraints])
#print(tt_influence(hypothesis, 0), tt_influence(hypothesis, 1), tt_influence(hypothesis, 2), tt_influence(hypothesis, 3))
print(get_CNF([head_coin_1, tail_coin_1, head_coin_2, tail_coin_2], hypothesis))
print([np.sum(np.abs(t[:, 0, :] - t[:, 1, :])) for t in hypothesis])
print("Score:", boolean_criterion(len(hypothesis))(hypothesis), tt_inner_prod(hypothesis, hypothesis))
print(f"Total time taken: {t_2-t_1}s.")