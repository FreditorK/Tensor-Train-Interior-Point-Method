import copy
import random
from typing import Dict

import numpy as np
from sympy.logic.boolalg import ANFform, to_cnf, to_dnf, to_anf
from sympy import symbols
from tt_op import *
from copy import deepcopy
from operators import partial_D, D_func
from abc import ABC, abstractmethod


class ParameterSpace(ABC):

    @property
    @abstractmethod
    def atom_count(self):
        ...

    @property
    @abstractmethod
    def atoms(self):
        ...

    @property
    @abstractmethod
    def hypothesis_count(self):
        ...

    @property
    @abstractmethod
    def hypotheses(self):
        ...


class Expression:

    def __init__(self, par_space: ParameterSpace, name: str, args, op):
        self.par_space = par_space
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
        if isinstance(self, Atom) and isinstance(other, Atom):
            return Boolean_Function(self.par_space, f"({self.name} ∧ {other.name})",
                                    tt_and(self.to_tt_train(), other.to_tt_train()))
        return Expression(self.par_space, f"({self.name} ∧ {other.name})", [self, other], tt_and)

    def __rand__(self, other):
        return other.__and__(self)

    def __or__(self, other):
        if isinstance(self, Atom) and isinstance(other, Atom):
            return Boolean_Function(self.par_space, f"({self.name} v {other.name})",
                                    tt_or(self.to_tt_train(), other.to_tt_train()))
        return Expression(self.par_space, f"({self.name} v {other.name})", [self, other], tt_or)

    def __ror__(self, other):
        return other.__or__(self)

    def __xor__(self, other):
        if isinstance(self, Atom) and isinstance(other, Atom):
            return Boolean_Function(self.par_space, f"({self.name} ⊻ {other.name})",
                                    tt_xor(self.to_tt_train(), other.to_tt_train()))
        return Expression(self.par_space, f"({self.name} ⊻ {other.name})", [self, other], tt_xor)

    def __rxor__(self, other):
        return other.__xor__(self)

    def __invert__(self):
        if isinstance(self, Atom):
            return Boolean_Function(self.par_space, f"¬{self.name}", tt_neg(self.to_tt_train()))
        return Expression(self.par_space, f"¬{self.name}", [self], tt_neg)

    def __lshift__(self, other):  # <-
        return self.__ror__(other.__invert__())

    def __rlshift__(self, other):
        return other.__ror__(self.__invert__())

    def __rshift__(self, other):  # ->
        return other.__ror__(self.__invert__())

    def __rrshift__(self, other):  # ->
        return self.__ror__(other.__invert__())


class Atom(Expression):

    def __init__(self, par_space: ParameterSpace, name=None):
        if name is None:
            name = f"a_{par_space.atom_count}"
        super().__init__(par_space, name, [self], lambda x: x)
        self.index = par_space.atom_count

    def __eq__(self, other):
        if isinstance(other, Atom):
            return self.index == other.index
        return False

    def to_tt_train(self):
        return tt_atom_train(self.index, self.par_space.atom_count + self.par_space.hypothesis_count)

    def __hash__(self):
        return self.index


class TTExpression:
    def __init__(self, cores: List[np.array], par_space: ParameterSpace, substituted=None):
        assert par_space.atom_count <= len(cores) <= (
            par_space.atom_count + par_space.hypothesis_count), "Labels do not match the tensor!"
        self.cores = cores
        self.par_space = par_space
        self.substituted = substituted
        if substituted is None:
            self.substituted = []

    @classmethod
    def from_expression(cls, expr: Expression):
        tt_train = expr.to_tt_train()
        not_involved_hypotheses_idxs = []
        for h in expr.par_space.hypotheses:
            # TODO: This condition might need a bit more thought
            if np.sum(
                np.abs(tt_train[h.index][:, 1, :])
            ) <= 1 / 2 ** (expr.par_space.atom_count + expr.par_space.hypothesis_count):
                not_involved_hypotheses_idxs.append(h.index)
        for i in sorted(not_involved_hypotheses_idxs, reverse=True):
            tt_train[i - 1] = np.einsum("ldr, rk -> ldk", tt_train[i - 1], tt_train[i][:, 0, :])
            tt_train.pop(i)
        return cls(tt_train, expr.par_space, substituted=not_involved_hypotheses_idxs)

    @property
    def hypotheses(self):
        return [h for h in self.par_space.hypotheses if h.index not in self.substituted]

    def __add__(self, other):
        if isinstance(other, TTExpression):
            assert self.par_space == other.par_space, "Tensors live in different parameter spaces!"
            return TTExpression(tt_add(self.cores, other.cores), self.par_space)
        new_cores = tt_mul_scal(other, tt_one(len(self.cores)))
        return TTExpression(tt_add(self.cores, new_cores), self.par_space)

    def __radd__(self, other):
        return other.__add__(self)

    def __sub__(self, other):
        if isinstance(other, TTExpression):
            assert self.par_space == other.par_space, "Tensors live in different parameter spaces!"
            new_cores = tt_mul_scal(-1, deepcopy(other.cores))
            return TTExpression(tt_add(self.cores, new_cores), self.par_space)
        new_cores = tt_mul_scal(-other, tt_one(len(self.cores)))
        return TTExpression(tt_add(self.cores, new_cores), self.par_space)

    def __rsub__(self, other):
        return other.__sub__(self)

    def __mul__(self, other):
        if isinstance(other, TTExpression):
            assert self.par_space == other.par_space, "Tensors live in different parameter spaces!"
            return TTExpression(tt_hadamard(self.cores, other.cores), self.par_space)
        new_cores = tt_mul_scal(other, deepcopy(self.cores))
        return TTExpression(new_cores, self.par_space)

    def __rmul__(self, other):
        return other.__mul__(self)

    def __and__(self, other):
        assert isinstance(other, TTExpression), "The AND-operation can only be performed between to TensorTrains!"
        assert self.par_space == other.par_space, "Tensors live in different parameter spaces!"
        return TTExpression(tt_and(self.cores, other.cores), self.par_space)

    def __rand__(self, other):
        return other.__and__(self)

    def __or__(self, other):
        assert isinstance(other, TTExpression), "The OR-operation can only be performed between to TensorTrains!"
        assert self.par_space == other.par_space, "Tensors live in different parameter spaces!"
        return TTExpression(tt_or(self.cores, other.cores), self.par_space)

    def __ror__(self, other):
        return other.__or__(self)

    def __xor__(self, other):
        assert isinstance(other, TTExpression), "The XOR-operation can only be performed between to TensorTrains!"
        assert self.par_space == other.par_space, "Tensors live in different parameter spaces!"
        return TTExpression(tt_xor(self.cores, other.cores), self.par_space)

    def __rxor__(self, other):
        return other.__xor__(self)

    def __invert__(self):
        new_cores = tt_mul_scal(-1, deepcopy(self.cores))
        return TTExpression(new_cores, self.par_space)

    def __neg__(self):
        new_cores = tt_mul_scal(-1, deepcopy(self.cores))
        return TTExpression(new_cores, self.par_space)

    def to_ANF(self):
        vocab = self.par_space.atoms + [h for h in self.par_space.hypotheses if h not in self.substituted]
        variable_names = " ".join([a.name for a in vocab])
        variables = symbols(variable_names)
        truth_table_labels = ((np.round(tt_to_tensor(tt_bool_op(self.cores))) + 1) / 2).astype(int).flatten()
        anf = ANFform(variables, list(reversed(truth_table_labels)))
        return anf

    def to_CNF(self):
        anf = self.to_ANF()
        return to_cnf(anf, simplify=True)

    def to_DNF(self):
        anf = self.to_ANF()
        return to_dnf(anf, simplify=True)


class Hypothesis(Expression):

    def __init__(self, par_space: ParameterSpace, name=None):
        if name is None:
            name = f"h_{par_space.hypothesis_count}"
        super().__init__(par_space, name, [self], lambda x: x)
        self.value = tt_leading_one(par_space.atom_count)
        self.index = par_space.hypothesis_count + par_space.atom_count

    def to_tt_train(self):
        return tt_atom_train(self.index, self.par_space.atom_count + self.par_space.hypothesis_count)

    def substitute_into(self, tt_train: TTExpression) -> TTExpression:
        assert self.par_space.atoms == tt_train.par_space.atoms, "Hypothesis is not in the parameter space of the given TT."
        assert self.par_space.hypotheses == tt_train.par_space.hypotheses, "Hypothesis is not in the hypothesis space of the given TT."
        index = self.index - sum([1 for i in tt_train.substituted if i < self.index])
        tt_core_without_basis = np.einsum("ldr, rk -> ldk", tt_train.cores[index - 1],
                                          tt_train.cores[index][:, 0, :])
        tt_core_with_basis = np.einsum("ldr, rk -> ldk", tt_train.cores[index - 1], tt_train.cores[index][:, 1, :])
        new_cores = tt_xnor(tt_train.cores[:index - 1] + [tt_core_with_basis] + tt_train.cores[index + 1:],
                            self.value + [np.array([1, 0]).reshape(1, 2, 1)] * self.par_space.hypothesis_count)
        new_cores = tt_rank_reduce(
            tt_add(tt_train.cores[:index - 1] + [tt_core_without_basis] + tt_train.cores[index + 1:], new_cores))
        return TTExpression(new_cores, self.par_space, substituted=tt_train.substituted + [self.index])

    def to_CNF(self):
        return TTExpression(self.value, self.par_space, substituted=self.par_space.hypotheses).to_CNF()

    def __eq__(self, other):
        if isinstance(other, Hypothesis):
            return self.index == other.index
        return False

    def __hash__(self):
        return self.index


class Boolean_Function(Expression):

    def __init__(self, par_space, name: str, tt_e):
        self.name = name
        self.tt_e = tt_e
        super().__init__(par_space, name, [self], lambda x: x)

    def to_tt_train(self):
        return self.tt_e


class LogicConstraint(ABC):
    def __init__(self, tt_expr: TTExpression, hypotheses_to_insert: List[Hypothesis], percent):
        self.tt_expr = tt_expr
        self.hypotheses_to_insert = hypotheses_to_insert
        self.percent = percent
        self.s_lower = 2 ** (-tt_expr.par_space.atom_count)

    def _get_hyperplane(self):
        tt_expr = deepcopy(self.tt_expr)
        for h in self.hypotheses_to_insert:
            tt_expr = h.substitute_into(tt_expr)
        bias = tt_leading_entry(tt_expr.cores) - self.percent
        last_normal_core = np.einsum("ldr, rk -> ldk", tt_expr.cores[-2], tt_expr.cores[-1][:, 1, :])
        normal = tt_expr.cores[:-2] + [last_normal_core]
        return normal, bias

    @abstractmethod
    def _projection(self, tt_h, tt_n, bias):
        ...

    def get_projection(self):
        normal, bias = self._get_hyperplane()
        if abs(bias) > 1:  # TODO: Then it is unsatisfiable, only happens in Multi-Hypothesis case
            return lambda tt_h: (tt_h, False)
        return lambda tt_h: self._projection(tt_h, normal, bias)

    @abstractmethod
    def is_satisfied(self, tt_h):
        ...


class ExistentialConstraint(LogicConstraint):

    def _projection(self, tt_h, tt_n, bias):
        tt_n = deepcopy(tt_n)
        func_result = tt_inner_prod(tt_h, tt_n)
        condition = func_result + bias <= self.s_lower
        if condition:
            tt_n = tt_mul_scal(1 / jnp.sqrt(tt_inner_prod(tt_n, tt_n)), tt_n)
            alpha = np.sqrt((1 - (bias - self.s_lower) ** 2) / (1 - func_result ** 2))
            beta = bias - self.s_lower + alpha * func_result
            tt_h = tt_add(tt_mul_scal(alpha, tt_h), tt_mul_scal(-beta, tt_n))
            tt_h = tt_rank_reduce(tt_h)
        return tt_h, condition

    def is_satisfied(self, tt_h):
        normal, bias = self._get_hyperplane()
        return tt_inner_prod(tt_h, normal) + bias > self.s_lower


class UniversalConstraint(LogicConstraint):

    def _projection(self, tt_h, tt_n, bias):
        tt_n = deepcopy(tt_n)
        func_result = tt_inner_prod(tt_h, tt_n)
        condition = abs(func_result + bias) >= self.s_lower
        if condition:
            tt_n = tt_mul_scal(1 / jnp.sqrt(tt_inner_prod(tt_n, tt_n)), tt_n)
            alpha = np.sqrt((1 - bias ** 2) / (1 - func_result ** 2))
            beta = bias + alpha * func_result
            tt_h = tt_add(tt_mul_scal(alpha, tt_h), tt_mul_scal(-beta, tt_n))
            tt_h = tt_rank_reduce(tt_h)
        return tt_h, condition

    def is_satisfied(self, tt_h):
        normal, bias = self._get_hyperplane()
        return abs(tt_inner_prod(tt_h, normal) + bias) < self.s_lower


class ConstraintSpace(ParameterSpace, ABC):
    def __init__(self):
        self.atom_list: List[Atom] = []
        self.hypothesis_list: List[Hypothesis] = []
        self.objectives: Dict[List] = dict()
        self.iq_constraints: Dict[List] = dict()
        self.eq_constraints: Dict[List] = dict()
        self.exclusions: Dict[List] = dict()

    def add_exclusion(self, hypothesis: Hypothesis):
        self.exclusions[hypothesis].append(tt_rank_reduce(tt_bool_op(hypothesis.value)))
    @property
    def atoms(self):
        return self.atom_list

    @property
    def atom_count(self):
        return len(self.atoms)

    @property
    def hypotheses(self):
        return self.hypothesis_list

    @property
    def permuted_hypotheses(self):
        return list(np.random.permutation(self.hypothesis_list))

    @property
    def hypothesis_count(self):
        return len(self.hypothesis_list)

    def generate_atoms(self, n):
        atoms = []
        k = self.atom_count
        for i in range(n):
            a = Atom(self, f"x_{k + i}")
            self.atom_list.append(a)
            atoms.append(a)
        return atoms

    def Atom(self, name=None):
        a = Atom(self, name)
        self.atoms.append(a)
        return a

    def Hypothesis(self, name=None):
        h = Hypothesis(self, name)
        self.hypotheses.append(h)
        self.iq_constraints[h] = []
        self.eq_constraints[h] = []
        self.exclusions[h] = [tt_mul_scal(-1, tt_one(self.atom_count))]
        return h

    def there_exists(self, expr: Expression):
        return self.at_least(expr, -1)

    def at_least(self, expr: Expression, percent):
        tt_train = TTExpression.from_expression(expr)
        relevant_hs = tt_train.hypotheses
        for i, h in enumerate(relevant_hs):
            self.iq_constraints[h].append(
                ExistentialConstraint(tt_train, relevant_hs[:i] + relevant_hs[i + 1:], percent))

    def for_all(self, expr: Expression, percent=1):
        tt_train = TTExpression.from_expression(expr)
        relevant_hs = tt_train.hypotheses
        for i, h in enumerate(relevant_hs):
            self.eq_constraints[h].append(
                UniversalConstraint(tt_train, relevant_hs[:i] + relevant_hs[i + 1:], percent))

    def project(self, hypothesis: Hypothesis):
        projections = [eq.get_projection() for eq in self.eq_constraints[hypothesis]] + [iq.get_projection() for iq in
                                                                                         self.iq_constraints[
                                                                                             hypothesis]]
        proj_tt_train = hypothesis.value
        not_converged = True
        while not_converged:
            not_converged = False
            for proj in projections:
                proj_tt_train, is_violated = proj(proj_tt_train)
                not_converged = not_converged or is_violated
        hypothesis.value = proj_tt_train

    def round(self, hypothesis: Hypothesis, error_bound):
        tt_train = hypothesis.value
        tt_table = tt_bool_op(tt_train)
        tt_table_p2 = tt_hadamard(tt_table, tt_table)
        tt_table_p3 = tt_hadamard(tt_table_p2, tt_table)
        if error_bound > 0:
            tt_table_p3 = self._add_repellers(tt_train, tt_table_p3, tt_table_p2, self.exclusions[hypothesis], error_bound)
        tt_update = tt_mul_scal(-0.5, tt_rank_reduce(tt_bool_op_inv(tt_table_p3)))
        tt_train = tt_mul_scal(1 - tt_inner_prod(tt_update, tt_train), tt_train)
        tt_train = tt_add(tt_train, tt_update)
        tt_train = tt_mul_scal(1 / jnp.sqrt(tt_inner_prod(tt_train, tt_train)), tt_train)
        hypothesis.value = tt_train

    def _add_repellers(self, tt_train, tt_table_p3, tt_table_p2, exclusions, error_bound):
        excl_to_apply = [excl for excl in exclusions if 1 - tt_inner_prod(tt_bool_op_inv(excl), tt_train) < error_bound]
        if len(excl_to_apply) > 0:
            repeller = excl_to_apply[0]
            for excl in excl_to_apply[1:]:
                repeller = tt_rank_reduce(tt_add(repeller, excl))
            repeller = tt_hadamard(tt_add(tt_one(self.atom_count), tt_mul_scal(-1, tt_table_p2)), repeller)
            tt_table_p3 = tt_rank_reduce(tt_add(tt_table_p3, repeller))
        return tt_table_p3

    def stopping_criterion(self, tt_trains, prev_tt_trains):
        return np.mean([
            1 - tt_inner_prod(tt_train, prev_tt_train) for tt_train, prev_tt_train in zip(tt_trains, prev_tt_trains)
        ])
