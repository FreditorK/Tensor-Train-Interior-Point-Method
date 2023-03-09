from utils import *

vocab_size = 3
x = Atom(vocab_size, "x")
y = Atom(vocab_size, "y")
z = Atom(vocab_size, "z")
e_3 = (
    (x << y) & (x << z)
)
e_tt = [2 * np.random.rand(1, 2, 1) - 1 for _ in range(3)] #e_3.to_tt_train()
e_tt[0] *= 1/np.sqrt(tt_inner_prod(e_tt, e_tt))
e_tt = tt_rl_orthogonalize(e_tt)
"""
print(tt_to_tensor(e_tt))
b = tt_bool_op(e_tt)
copy_b = deepcopy(b)
print("ha", tt_to_tensor(b))
b[0] *= np.round(b[0]/np.abs(b[0]), decimals=1)
print(tt_to_tensor(b))
abs_b = [1/b[0]] + [np.round(t/np.abs(t), decimals=1) for t in b[1:]]
#abs_b = tt_bool_op_inv(abs_b)
abs_b = tt_hadamard(copy_b, abs_b)
print("Haha", tt_to_tensor(abs_b))
ak = [np.array([1/np.sqrt(5), 2/np.sqrt(5)]).reshape(1, 2, 1) for _ in range(20)]
print(tt_inner_prod(ak, ak))
#a = tt_bool_op_inv(b)
#print(tt_to_tensor(a))
"""
"""
for _ in range(20):
    #e_tt = tt_bool_op(e_tt)
    e_tt_copy = deepcopy(e_tt)
    one = tt_one(len(e_tt))
    sq_e_tt = tt_hadamard(e_tt, e_tt)
    sq_e_tt[0] *= -1
    e_tt = tt_hadamard(tt_add(one, sq_e_tt), e_tt)
    e_tt[0] *= 0.2
    e_tt = tt_add(e_tt_copy, e_tt)
    e_tt = tt_rl_orthogonalize(e_tt)
    #norm
    #e_tt = tt_bool_op_inv(e_tt)
    #e_tt[0] = e_tt[0] / jnp.sqrt(tt_inner_prod(e_tt, e_tt))
    #e_tt = tt_rl_orthogonalize(e_tt)
    print(tt_to_tensor(e_tt))

#print(tt_to_tensor(tt_bool_op(e_tt)))
"""
e= np.array([0.2, 0.01, -0.5, 0.6])#4*np.random.rand(4) - 2
beta = -0.5
print(e)
a = 1
for _ in range(10):
    e = e - beta*(1-e**2)*e #- (1/a)*np.array([0.0, 0.0, 0.5, 1.0]).sum()
    a += 1
    print(e)