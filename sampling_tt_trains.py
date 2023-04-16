import numpy as np
from tt_op import *
import matplotlib.pyplot as plt
import copy

length = 8
tt_train = [2*np.random.randint(0, 2, size=(1, 2, 1))-1 for _ in range(length-1)] + [2*np.random.randint(0, 2, size=(1, 2, 1))-1]
iterations = 10
for i in range(1, length-1):
    tt_train = tt_add(tt_train, [np.ones((1, 2, 4))] + [np.ones((1, 2, 4)) if j != i else np.random.randn(4, 2, 4) for j in range(1, length)] + [np.ones((4, 2, 1))])
    i = np.random.randint(length)
    cop = copy.deepcopy(tt_train[i])
    cop.at[:, 0].set(tt_train[i][:, 1])
    cop.at[:, 1].set(tt_train[i][:, 0])
    tt_train[i] = cop
    tt_train = tt_rank_reduce(tt_train)
tt_train[0] *= 1/np.sqrt(iterations)
number = 2**length

data_prod = tt_to_tensor(tt_train).flatten()

data_sing = np.random.randn(number)
print([c.shape for c in tt_train])
print(tt_to_tensor(tt_train))

plt.hist(data_prod, bins=(np.arange(24)-12)/4, ec="k", label="double")
plt.hist(data_sing, bins=(np.arange(24)-12)/4, alpha=0.5, label="single")

plt.show()

