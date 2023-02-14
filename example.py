from time import time
from utils import *
from optimiser import Minimiser

vocab_size = 3
x = Atom(3, "x")
y = Atom(3, "y")
z = Atom(3, "z")
constraints = Constraint()
constraints.all_A_extending(y & z)
constraints.exists_A_extending(x << (y & z))
opt = Minimiser(constraints, vocab_size)
t_1 = time()
hypothesis = opt.find_feasible_hypothesis()
t_2 = time()
print(get_CNF([x, y, z], hypothesis))
print(np.round(tt_to_tensor(hypothesis), decimals=5))
print("Score:", boolean_criterion(len(hypothesis))(hypothesis), tt_inner_prod(hypothesis, hypothesis))
print(f"Total time taken: {t_2-t_1}s.")