from time import time
from utils import *
from optimiser import Minimiser


head_coin_1 = Atom(4, "head_coin_1")
tail_coin_1 = Atom(4, "tail_coin_1")
head_coin_2 = Atom(4, "head_coin_2")
tail_coin_2 = Atom(4, "tail_coin_2")
opt = Minimiser([
    all_A_extending(
        (head_coin_1 & ~tail_coin_1) |
        (~head_coin_1 & tail_coin_1) &
        (head_coin_2 & ~tail_coin_2) |
        (~head_coin_2 & tail_coin_2)
    )
], 4)
t_1 = time()
hypothesis = opt.find_feasible_hypothesis()
t_2 = time()
#print(tt_influence(hypothesis, 0), tt_influence(hypothesis, 1), tt_influence(hypothesis, 2), tt_influence(hypothesis, 3))
print(get_ANF([head_coin_1, tail_coin_1, head_coin_2, tail_coin_2], hypothesis))
print([np.sum(np.abs(t[:, 0, :] - t[:, 1, :])) for t in hypothesis])
print(np.round(tt_to_tensor(hypothesis), decimals=5))
print("Score:", boolean_criterion(len(hypothesis))(hypothesis), tt_inner_prod(hypothesis, hypothesis))
print(f"Total time taken: {t_2-t_1}s.")