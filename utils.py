import numpy as np
import jax.numpy as jnp
import jax.scipy.special as sc
from numbers import Number
from abc import ABC, abstractmethod
from typing import Dict, Callable, List
import math
from tt_op import *
from copy import deepcopy


class Expression:
    count = 0

    def __init__(self, name: str, args, op):
        if name is None:
            self.name = f"e_{str(Expression.count)}"
            Expression.count += 1
        else:
            self.name = name

        self.op = op
        self.args = args

    def to_tt_train(self):
        return self.op(*[a.to_tt_train() for a in self.args])

    def __repr__(self):
        return str(self)

    def __str__(self):
        return self.name

    def __getitem__(self, key):
        return self.args[key]

    def __and__(self, other):
        return Expression(f"({self.name} ∧ {other.name})", [self, other], tt_and)

    def __rand__(self, other):
        return other.__and__(self)

    def __or__(self, other):
        return Expression(f"({self.name} v {other.name})", [self, other], tt_or)

    def __ror__(self, other):
        return other.__or__(self)

    def __xor__(self, other):
        return Expression(f"({self.name} ⊻ {other.name})", [self, other], tt_xor)

    def __rxor__(self, other):
        return other.__or__(self)

    def __invert__(self):
        return Expression(f"¬{self.name}", [self], tt_neg)

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

    def __init__(self, vocab_size, name=None):
        if name is None:
            name = f"v_{str(Atom.counter)}"
        super().__init__(name, [self], lambda x: x)
        self.index = Atom.counter
        self.tt_train = tt_atom_train(self.index, vocab_size)
        Atom.counter += 1

    def to_tt_train(self):
        return deepcopy(self.tt_train)


def exists_A_extending(e):
    e_mean = tt_leading_entry(e) + 1
    return lambda h: -(tt_leading_entry(h) + e_mean + tt_inner_prod(h, e)) + 1e-6


def exists_A_not_extending(e):
    e_mean = tt_leading_entry(e) - 1
    return lambda h: tt_leading_entry(h) + e_mean + tt_inner_prod(h, e) + 1e-6


def all_A_extending(e):
    e_mean = tt_leading_entry(e) - 1
    return lambda h: -(tt_leading_entry(h) + e_mean + tt_inner_prod(h, e))


def all_A_not_extending(e):
    e_mean = tt_leading_entry(e) + 1
    return lambda h: tt_leading_entry(h) + e_mean + tt_inner_prod(h, e)
