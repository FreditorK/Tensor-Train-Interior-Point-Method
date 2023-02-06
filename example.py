from time import time
from utils import *
from optimiser import Minimiser

x = Atom(3, "x")
y = Atom(3, "y")
z = Atom(3, "z")
opt = Minimiser([
    exists_A_extending(x << (y & z)),
    all_A_not_extending(y ^ z)
], 3)
t_1 = time()
hypothesis = opt.find_feasible_hypothesis(100)
t_2 = time()
print(get_ANF([x, y, z], hypothesis))
print(np.round(tt_to_tensor(hypothesis), decimals=5))
print("Score:", boolean_criterion(hypothesis), tt_inner_prod(hypothesis, hypothesis))
print(f"Total time taken: {t_2-t_1}s.")