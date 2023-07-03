from time import time
from utils import *
from optimiser import ILPSolver, AnswerSetSolver

vocab_size = 4
head_coin_1 = Atom(vocab_size, "head_coin_1")
tail_coin_1 = Atom(vocab_size, "tail_coin_1")
head_coin_2 = Atom(vocab_size, "head_coin_2")
tail_coin_2 = Atom(vocab_size, "tail_coin_2")
h = Hypothesis()


def coin_symmetry(tt_train):
    return tt_leading_entry(tt_train) #tt_shared_influence(tt_train, np.array([0, 1])) #jnp.abs(tt_shared_influence(tt_train, np.array([0, 1])) - tt_shared_influence(tt_train, np.array([2, 3])))


const_space = ConstraintSpace(vocab_size)
e_0 = Boolean_Function(head_coin_1 ^ tail_coin_1)
e_1 = Boolean_Function(head_coin_2 ^ tail_coin_2)
const_space.forall_S(h >> e_0)
const_space.forall_S(h >> e_1)
opt = ILPSolver(const_space, vocab_size)
t_1 = time()
hypothesis = opt.find_feasible_hypothesis()
t_2 = time()
print(tt_leading_entry(hypothesis), tt_shared_influence(hypothesis, np.array([0, 1])), tt_shared_influence(hypothesis, np.array([2, 3])))
print("Shapes: ", [t.shape for t in hypothesis])
print("Equality constraint Score: ", [jnp.sum(jnp.abs(c(hypothesis))) for c in const_space.eq_constraints])
print("Inequality constraint Score: ", [jnp.sum(c(hypothesis)) for c in const_space.iq_constraints])
print(get_CNF([head_coin_1, tail_coin_1, head_coin_2, tail_coin_2], hypothesis))
print("Score:", tt_boolean_criterion(len(hypothesis))(hypothesis), tt_inner_prod(hypothesis, hypothesis))
print(f"Total time taken: {t_2 - t_1}s.")
asp_solver = AnswerSetSolver([head_coin_1, tail_coin_1, head_coin_2, tail_coin_2])
X = asp_solver.get_minimal_answer_set(hypothesis, head_coin_1=1)
print(X)
