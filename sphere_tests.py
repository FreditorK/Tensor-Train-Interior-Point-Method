import numpy as np
from tt_op import *
from scipy.stats import shapiro
from utils import *

tensor = 2*np.random.rand(*[2 for _ in range(4)])-1

tt_tensor = tt_svd(tensor)
tt_tensor[0] *= 1/np.sqrt(tt_inner_prod(tt_tensor, tt_tensor))

tt_table = tt_bool_op(tt_tensor)
flattened_tensor = tt_to_tensor(tt_table).flatten()
flattened_tensor = np.sort(flattened_tensor)
print(flattened_tensor)
stat, p = shapiro(flattened_tensor)
print('Statistics=%.3f, p=%.3f' % (stat, p))
# interpret
alpha = 0.05
if p > alpha:
    print('Sample looks Gaussian (fail to reject H0)')
else:
    print('Sample does not look Gaussian (reject H0)')
results = []
a = tt_to_tensor(tt_tensor)
a[0, 0, 0, 0] = 0
tt_tensor = tt_svd(a)
for _ in range(20):
    input_x = [np.array([1.0, 2 * np.random.randint(0, 2) - 1]).reshape(1, 2, 1) for _ in range(4)]
    results.append(tt_inner_prod(tt_tensor, input_x))

stat, p = shapiro(np.array(results))
print('Statistics=%.3f, p=%.3f' % (stat, p))
# interpret
alpha = 0.05
if p > alpha:
    print('Sample looks Gaussian (fail to reject H0)')
else:
    print('Sample does not look Gaussian (reject H0)')

"""
vocab_size = 4
x = Atom(vocab_size, "x")
y = Atom(vocab_size, "y")
z = Atom(vocab_size, "z")
w = Atom(vocab_size, "w")
tt_tensor = (x & (y | z | w)).to_tt_train()
print(tt_to_tensor(tt_bool_op(tt_tensor)))
big_tensor = tt_to_tensor(tt_tensor)
print(big_tensor[0, 0, 0, 0] )
big_tensor[0, 0, 0, 0] *= (big_tensor[0, 0, 0, 0]+1)
tt_tensor = tt_svd(big_tensor)
print(tt_to_tensor(tt_bool_op(tt_noise_op(tt_tensor, np.array([-1.0, -1.0, -1.0, -1.0])))))
"""