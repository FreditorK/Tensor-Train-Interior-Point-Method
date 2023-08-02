from utils import *

const_space = ConstraintSpace()
x = const_space.Atom("x")
y = const_space.Atom("y")
h = const_space.Hypothesis("h_0")
const_space.for_all(~h | (x & y))
const_space.project(h)