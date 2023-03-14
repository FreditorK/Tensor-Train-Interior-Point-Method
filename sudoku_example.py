from time import time

from optimiser import AnswerSetSolver, Minimiser
from utils import *

"""
sudoku_e_1 = np.array(
    [
        [None, None, 3, None],
        [None, None, 4, None],
        [None, 4, None, None],
        [None, 1, None, None]
    ]
)

sudoku_neg_e_1 = np.array(
    [
        [1, None, 1, None],
        [None, None, None, None],
        [None, None, None, None],
        [None, None, None, None]
    ]
)

sudoku_neg_e_2 = np.array(
    [
        [1, None, None, None],
        [None, None, None, None],
        [1, None, None, None],
        [None, None, None, None]
    ]
)

sudoku_neg_e_3 = np.array(
    [
        [1, None, None, None],
        [None, 1, None, None],
        [None, None, None, None],
        [None, None, None, None]
    ]
)
"""
vocab_size = 6
entry_small = Atom(vocab_size, name="entry_small")
entry_odd = Atom(vocab_size, name="entry_odd")
column_small = Atom(vocab_size, name="column_small")
column_odd = Atom(vocab_size, name="column_odd")
row_small = Atom(vocab_size, name="row_small")
row_odd = Atom(vocab_size, name="row_odd")

# observation
e_1 = Boolean_Function(
    (~column_small & column_odd & row_small & row_odd & ~entry_small & entry_odd) |
    (~column_small & column_odd & row_small & ~row_odd & ~entry_small & ~entry_odd) |
    (column_small & ~column_odd & ~row_small & row_odd & ~entry_small & ~entry_odd) |
    (column_small & ~column_odd & ~row_small & ~row_odd & entry_small & entry_odd)
)

# negative observations

neg_e_1 = Boolean_Function(
    ~(
        (column_small & column_odd & row_small & row_odd & entry_small & entry_odd) |
        (~column_small & column_odd & row_small & row_odd & entry_small & entry_odd)
    )
)

neg_e_2 = Boolean_Function(
    ~(
        (column_small & column_odd & row_small & row_odd & entry_small & entry_odd) |
        (column_small & column_odd & ~row_small & row_odd & entry_small & entry_odd)
    )
)

neg_e_3 = Boolean_Function(
    ~(
        (column_small & column_odd & row_small & row_odd & entry_small & entry_odd) |
        (column_small & ~column_odd & row_small & ~row_odd & entry_small & entry_odd)
    )
)

h = Hypothesis()
const_space = ConstraintSpace()
const_space.exists_S(h >> e_1)
const_space.forall_S(h >> neg_e_1)
const_space.forall_S(h >> neg_e_2)
const_space.forall_S(h >> neg_e_3)
opt = Minimiser(const_space, vocab_size)
t_1 = time()
hypothesis = opt.find_feasible_hypothesis()
t_2 = time()
print("Ranks: ", [t.shape for t in hypothesis])
print("Equality constraint Score: ", [jnp.sum(jnp.abs(c(hypothesis))) for c in const_space.eq_constraints])
print("Inequality constraint Score: ", [jnp.sum(c(hypothesis)) for c in const_space.iq_constraints])
print("Score:", boolean_criterion(len(hypothesis))(hypothesis), tt_inner_prod(hypothesis, hypothesis))
print(f"Total time taken: {t_2-t_1}s.")
#asp_solver = AnswerSetSolver(hypothesis, atoms=[entry_small, entry_odd, column_small, column_odd, row_small, row_odd])
#X = asp_solver.get_answer_set()
#print(X)