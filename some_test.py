import numpy as np

from operators import D_func
from utils import *
from tt_op import tt_extract_seq

vocab_size = 3
x = Atom(vocab_size, "x")
y = Atom(vocab_size, "y")
z = Atom(vocab_size, "z")
e_3 = (
    x & (y | z) #~x & (y | z)
)
e_tt = e_3.to_tt_train()

print(tt_extract_seq(e_tt, dict([(0, 1), (2, -1)])))
"""
tt_ttable = tt_bool_op(e_tt)
#mask = tt_add(tt_add(x.tt_train, y.tt_train), z.tt_train)
#tt_ttable = tt_hadamard(tt_ttable, mask)
tt_ttable = tt_rl_orthogonalize(tt_ttable)
#tt_ttable[0] *= 1/tt_inner_prod(tt_ttable, tt_ttable)
print("hi", tt_to_tensor(tt_ttable))
#print(tt_inner_prod(tt_ttable, [np.array([0.1, 0.9]).reshape(1, 2, 1), np.array([0.1, 0.9]).reshape(1, 2, 1), np.array([0.1, 0.9]).reshape(1, 2, 1)]))
#print(tt_inner_prod(tt_ttable, [np.array([0.9, 0.1]).reshape(1, 2, 1), np.array([0.1, 0.9]).reshape(1, 2, 1), np.array([0.1, 0.9]).reshape(1, 2, 1)]) - tt_inner_prod(tt_ttable, [np.array([0.1, 0.9]).reshape(1, 2, 1), np.array([0.1, 0.9]).reshape(1, 2, 1), np.array([0.1, 0.9]).reshape(1, 2, 1)]))
#print(tt_inner_prod(tt_ttable, [np.array([0.9, 0.1]).reshape(1, 2, 1), np.array([0.9, 0.1]).reshape(1, 2, 1), np.array([0.1, 0.9]).reshape(1, 2, 1)]) - tt_inner_prod(tt_ttable, [np.array([0.1, 0.9]).reshape(1, 2, 1), np.array([0.1, 0.9]).reshape(1, 2, 1), np.array([0.1, 0.9]).reshape(1, 2, 1)]))
#print(tt_inner_prod(tt_ttable, [np.array([1.0, 0.1]).reshape(1, 2, 1), np.array([1.0, 0.1]).reshape(1, 2, 1), np.array([1.0, 0.1]).reshape(1, 2, 1)]))
#print(tt_inner_prod(tt_ttable, [np.array([0.9, 0.1]).reshape(1, 2, 1), np.array([0.9, 0.1]).reshape(1, 2, 1), np.array([0.9, 0.1]).reshape(1, 2, 1)]) - tt_inner_prod(tt_ttable, [np.array([0.1, 0.9]).reshape(1, 2, 1), np.array([0.9, 0.1]).reshape(1, 2, 1), np.array([0.9, 0.1]).reshape(1, 2, 1)]))

answer = [np.array([0.1, 0.9]).reshape(1, 2, 1) for _ in range(3)]
score = tt_inner_prod(tt_ttable, answer)
indices = [0, 1, 2]
for i in range(3):
    improvements = -np.ones(3)
    buffer = deepcopy(answer)
    for j in range(3):
        if j in indices:
            buffer[j] = np.array([0.9, 0.1]).reshape(1, 2, 1)
            improvements[j] = tt_inner_prod(tt_ttable, buffer) - score
            buffer[j] = answer[j]
    max_improvements = np.argmax(improvements)
    print(improvements, max_improvements)
    answer[max_improvements] = np.array([1.0, 0.0]).reshape(1, 2, 1)
    indices.remove(max_improvements)
    score = tt_inner_prod(tt_ttable, answer)
    if score >= 0.9**2:
        break
print("Halo", [(np.array([1, -1]) @ np.round(a)).item() for a in answer])

answer_set = np.array([-1, -1, -1])
x_num = tt_inner_prod(tt_ttable, [np.array([1.0, 0.0]).reshape(1, 2, 1), np.array([1.0, -1.0]).reshape(1, 2, 1), np.array([1.0, -1.0]).reshape(1, 2, 1)])
#print(x_num)

y_num = tt_inner_prod(tt_ttable, [np.array([1.0, -1.0]).reshape(1, 2, 1), np.array([1.0, 0.0]).reshape(1, 2, 1), np.array([1.0, -1.0]).reshape(1, 2, 1)])
#print(y_num)
z_num = tt_inner_prod(tt_ttable, [np.array([1.0, -1.0]).reshape(1, 2, 1), np.array([-1.0, -1.0]).reshape(1, 2, 1), np.array([1.0, 0.0]).reshape(1, 2, 1)])
#print(z_num)
x_num = tt_inner_prod(tt_ttable, [np.array([1.0, 0.0]).reshape(1, 2, 1), np.array([-1.0, -1.0]).reshape(1, 2, 1), np.array([1.0, -1.0]).reshape(1, 2, 1)])
#print(x_num)
#y_num = tt_inner_prod(tt_ttable, [np.array([-1.0, -1.0]).reshape(1, 2, 1), np.array([1.0, 0.0]).reshape(1, 2, 1), np.array([1.0, -1.0]).reshape(1, 2, 1)])
#print(y_num)
z_num = tt_inner_prod(tt_ttable, [np.array([-1.0, -1.0]).reshape(1, 2, 1), np.array([-1.0, -1.0]).reshape(1, 2, 1), np.array([1.0, 0.0]).reshape(1, 2, 1)])
#print(z_num)
def tt_train_x(answer_set, a_idx):
    tt_train = [np.array([1.0, -1.0]).reshape(1, 2, 1) for _ in range(len(answer_set))]
    for i, a in enumerate(answer_set):
        if i==a_idx:
            tt_train[i][:, 1] = 0.0
        elif a > 0:
            tt_train[i][:, 1] = -1.0
    return tt_train


def get_answer_set(tt_example, *atoms, itertions=10):
    mask = atoms[0].tt_train
    for a in atoms[1:0]:
        mask = tt_add(mask, a.tt_train)
        mask = tt_rl_orthogonalize(mask)
    tt_ttable = tt_hadamard(tt_bool_op(tt_example), mask)
    tt_ttable = tt_rl_orthogonalize(tt_ttable)
    tt_ttable[0] *= 1/tt_inner_prod(tt_ttable, tt_ttable)
    N = len(atoms)
    answer_set = -np.ones(N)
    for i in range(itertions):
        for a_idx in range(N):
            answer_set[a_idx] = tt_inner_prod(tt_ttable, tt_train_x(answer_set, a_idx))
    return answer_set


print(get_answer_set(e_tt, x, y, z))

print(tt_influence(e_tt, 0), tt_influence(e_tt, 1), tt_influence(e_tt, 2))

idxs = np.argsort([tt_shared_influence(e_tt, 0, 1), tt_shared_influence(e_tt, 0, 2), tt_shared_influence(e_tt, 1, 2)])
answer_set = np.array([-1.0, -1.0, -1.0])
gradient = D_func(lambda x: tt_inner_prod(e_tt, x))
for _ in range(5):
    for i in idxs:
        x_0 = [np.array([1.0, answer_set[i]]).reshape(1, 2, 1) if j != i else np.array([1.0, 0.0]).reshape(1, 2, 1) for j in range(3)]
        x = [np.array([1.0, answer_set[j]]).reshape(1, 2, 1) for j in range(3)]
        grad = gradient(x)[i].flatten()[-1]
        if grad != 0:
            answer_set[i] = np.sign((1-tt_inner_prod(e_tt, x_0))/grad)
        elif 1- tt_inner_prod(e_tt, x) > 1e-4:
            answer_set[:i] *= -1
            answer_set[i+1:] *= -1
        else:
            break
print(answer_set)

answer_set = np.array([-1, -1, -1])
prob_of_change = np.zeros(3)
for i, a in enumerate(answer_set):
    total_influence = tt_influence(e_tt, i)
    prob_of_change[i] = total_influence - sum([tt_shared_influence(e_tt, i, j) for j, s in enumerate(answer_set) if j != i])
idxs = np.argsort(prob_of_change)[::-1]
for i in idxs:
    if 1 - tt_inner_prod(e_tt, [np.array([1, a]).reshape(1, 2, 1) for a in answer_set]) < 1e-5:
        break
    else:
        answer_set[i] = -answer_set[i]

answer_set = np.array([-1, -1])
buffer = np.array([-1, -1])
for _ in range(10):
    answer_set[0] = np.sign((2)*(1 + 0.5 - 0.5*answer_set[1] + 0.5*answer_set[0]*answer_set[1]))
    answer_set[1] = np.sign((2)*(1 + 0.5 - 0.5 * answer_set[0] + 0.5 * answer_set[0] * answer_set[1]))
    print(answer_set)

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

e= np.array([0.2, 0.01, -0.5, 0.6])#4*np.random.rand(4) - 2
beta = -0.5
print(e)
a = 1
for _ in range(10):
    e = e - beta*(1-e**2)*e #- (1/a)*np.array([0.0, 0.0, 0.5, 1.0]).sum()
    a += 1
"""