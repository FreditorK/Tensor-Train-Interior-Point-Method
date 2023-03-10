from utils import *
from tt_op import *

head = Predicate(2, name="head")
tail = Predicate(2, name="tail")
x = Atom(2, "x")
y = Atom(2, "y")
#e = Boolean_Function(head(x) << tail(x))
#print(tt_to_tensor(e.to_tt_constraint()[-1]))
print(tt_to_tensor((head(x)).to_tt_train()),
      tt_to_tensor((tail(y)).to_tt_train()))
print(tt_to_tensor((head(x) & tail(y)).to_tt_train()))