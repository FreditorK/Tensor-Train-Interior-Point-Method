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
