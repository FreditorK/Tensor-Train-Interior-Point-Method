import numpy as np
import jax.numpy as jnp
import jax.scipy.special as sc
from numbers import Number
from abc import ABC, abstractmethod
from typing import Dict, Callable, List
import math


class Expression:
    count = 0

    def __init__(self, name: str, op):
        if name is None:
            self.name = f"e_{str(Expression.count)}"
            Expression.count += 1
        else:
            self.name = name

        self.tt_train = []
        self.op = op

    def __repr__(self):
        return str(self)

    def __str__(self):
        return self.name

    def __and__(self, other):
        pass

    def __rand__(self, other):
        return other.__and__(self)

    def __or__(self, other):
        pass

    def __ror__(self, other):
        return other.__or__(self)

    def __invert__(self):
        self.tt_train[0] *= -1
        return self

    def __lshift__(self, other):  # <-
        return self.__ror__(other.__invert__())

    def __rlshift__(self, other):
        return other.__ror__(self.__invert__())

    def __rshift__(self, other):  # ->
        return other.__ror__(self.__invert__())

    def __rrshift__(self, other):  # ->
        return self.__ror__(other.__invert__())


class Atom(Expression):
    counter = 0

    def __init__(self, index, name=None):
        if name is None:
            name = f"v_{str(Atom.counter)}"
        Atom.counter += 1
        super().__init__("{" + name + "}", [self], lambda x: x)
        self.index = Atom.counter
