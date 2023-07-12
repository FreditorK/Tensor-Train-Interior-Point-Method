from time import time
from optimiser import AnswerSetSolver, ILPSolver
from utils import *
import jax.numpy as jnp

vocab_size = 3
x = Atom(vocab_size, "x")
y = Atom(vocab_size, "y")
z = Atom(vocab_size, "z")
func = (x & y & z).to_tt_train()
a = tt_to_tensor(tt_bool_op(func))
print(a)
b = jnp.sqrt(jnp.linalg.norm(a.reshape(4, 2), ord=2))
print(b)
a = tt_to_tensor(tt_bool_op(tt_add_noise(func, noise_radius=0.6)))
print(a)
b = jnp.sqrt(jnp.linalg.norm(a.reshape(4, 2), ord=2))
print(b)
#print(tt_shared_influence(func, np.array([0, 1, 2])))