from utils import *
from tt_op import *

head = Predicate(2, name="head")
tail = Predicate(2, name="tail")
x = Atom(2, "x")
y = Atom(2, "y")
e = Boolean_Function(head(x) << tail(x))
print(tt_to_tensor(e.to_tt_constraint()[-1]))