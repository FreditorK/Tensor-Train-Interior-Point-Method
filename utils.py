import numpy as np
import jax.numpy as jnp
import jax.scipy.special as sc
from numbers import Number
from abc import ABC, abstractmethod
from typing import Dict, Callable, List
import math


class Expression:
    count = 0

    def __init__(self, name: str, values, op):
        if name is None:
            self.name = f"e_{str(Expression.count)}"
            Expression.count += 1
        else:
            self.name = name

        self.values = values
        self.op = op

    def forward(self, var_dict):
        return self.op(*[v if isinstance(v, Number) or isinstance(v, jnp.DeviceArray) else v.forward(var_dict) for v in
                         self.values])

    def to_tt_train (self):

    def __repr__(self):
        return str(self)

    def __str__(self):
        return self.name

    def __add__(self, other):  # +
        if isinstance(other, Number):
            other = float(other) + 0.0
        return Expression(f"({str(self)}+{str(other)})", [self, other], lambda x, y: x + y)

    def __sub__(self, other):  # -
        if isinstance(other, Number):
            other = float(other) + 0.0
        return Expression(f"({str(self)}-{str(other)})", [self, other], lambda x, y: x - y)

    def __mul__(self, other):  # *
        if isinstance(other, Number):
            other = float(other) + 0.0
        return Expression(f"{str(self)}*{str(other)}", [self, other], lambda x, y: x * y)

    def __neg__(self):  # -x
        return Expression(f"-{str(self)}", [self], lambda x: -x)

    # reverse math
    def __radd__(self, other):  # +
        if isinstance(other, Number):
            other = float(other) + 0.0
        return Expression(f"({str(other)}+{str(self)})", [self, other], lambda x, y: x + y)

    def __rsub__(self, other):  # -
        if isinstance(other, Number):
            other = float(other) + 0.0
        return Expression(f"({str(other)}-{str(self)})", [self, other], lambda x, y: y - x)

    def __rmul__(self, other):  # *
        if isinstance(other, Number):
            other = float(other) + 0.0
        return Expression(f"{str(other)}*{str(self)}", [self, other], lambda x, y: x * y)

    def __matmul__(self, other):
        return Expression(f"{str(other)}@{str(self)}", [self, other], lambda x, y: x @ y)

    def __rmatmul__(self, other):
        return Expression(f"{str(other)}@{str(self)}", [self, other], lambda x, y: y @ x)

    def __and__(self, other):
        return Expression(f"({str(self)}∧{str(other)})", [self, other],
                          lambda x, y: -0.5 + 0.5 * x + 0.5 * y + 0.5 * x * y)

    def __rand__(self, other):
        return Expression(f"({str(other)}∧{str(self)})", [self, other],
                          lambda x, y: -0.5 + 0.5 * y + 0.5 * x + 0.5 * y * x)

    def __or__(self, other):
        return Expression(f"({str(self)}v{str(other)})", [self, other],
                          lambda x, y: 0.5 + 0.5 * x + 0.5 * y - 0.5 * x * y)

    def __ror__(self, other):
        return Expression(f"({str(other)}v{str(self)})", [self, other],
                          lambda x, y: 0.5 + 0.5 * y + 0.5 * x - 0.5 * y * x)

    def __invert__(self):
        return Expression(f"~{str(self)}", [self], lambda x: -x)

    def __lshift__(self, other):  # <-
        return Expression(f"({str(self)}<-{str(other)})", [self, other],
                          lambda x, y: 0.5 + 0.5 * x - 0.5 * y + 0.5 * x * y)

    def __rlshift__(self, other):
        return Expression(f"({str(other)}<-{str(self)})", [self, other],
                          lambda x, y: 0.5 + 0.5 * y - 0.5 * x + 0.5 * y * x)

    def __rshift__(self, other):  # ->
        return Expression(f"({str(self)}->{str(other)})", [self, other],
                          lambda x, y: 0.5 - 0.5 * y + 0.5 * x + 0.5 * y * x)

    def __rrshift__(self, other):  # ->
        return Expression(f"({str(other)}->{str(self)})", [self, other],
                          lambda x, y: 0.5 - 0.5 * x + 0.5 * y + 0.5 * x * y)


class Atom(Expression):
    counter = 0

    def __init__(self, index, name=None):
        if name is None:
            name = f"v_{str(Atom.counter)}"
            Atom.counter += 1
        super().__init__("{" + name + "}", [self], lambda x: x)
        self.index = index

    def forward(self, var_dict: Dict[str, np.array]):
        return var_dict[self.name[1:-1]]
