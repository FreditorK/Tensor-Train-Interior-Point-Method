import copy
import random
from typing import Dict

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
            name = f"a_{str(par_space.atom_count)}"
        super().__init__(par_space, name, [self], lambda x: x)
        self.index = par_space.atom_count

    def to_tt_train(self):
        return tt_atom_train(self.index, self.par_space.atom_count + self.par_space.hypothesis_count)


class TTExpression:
    def __init__(self, cores: List[np.array], par_space: ParameterSpace, substituted=None):
        assert par_space.atom_count <= len(cores) <= (par_space.atom_count + par_space.hypothesis_count), "Labels do not match the tensor!"
        self.cores = cores
        self.par_space = par_space
        self.substituted = substituted
        if substituted is None:
            self.substituted = []

    @classmethod
    def from_expression(cls, expr: Expression):
        tt_train = expr.to_tt_train()
        return cls(tt_train, expr.par_space)

    @property
    def hypotheses(self):
        involved_hypotheses = []
        for h in self.par_space.hypotheses:
            if np.sum(self.cores[h.index][:, 1, :]) < 1e-8:
                involved_hypotheses.append(h)
        return involved_hypotheses

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
        vocab = self.par_space.atoms + [h for h in self.par_space.hypotheses if h.index not in self.substituted]
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
        assert self.par_space == tt_train.par_space, "Hypothesis is not in the parameter space of the given TT."
        index = self.index - sum([1 for i in tt_train.substituted if i < self.index])
        tt_core_without_basis = np.einsum("ldr, rk -> ldk", tt_train.cores[index - 1], tt_train.cores[index][:, 0, :]) # [self.index - 1]
        tt_core_with_basis = np.einsum("ldr, rk -> ldk", tt_train.cores[index - 1], tt_train.cores[index][:, 1, :])
        new_cores = tt_xnor(tt_train.cores[:index-1] + [tt_core_with_basis] + tt_train.cores[index+1:], self.value + [np.array([1, 0]).reshape(1, 2, 1)] * self.par_space.hypothesis_count)
        new_cores = tt_rank_reduce(tt_add(tt_train.cores[:index-1] + [tt_core_without_basis] + tt_train.cores[index+1:], new_cores))
        return TTExpression(new_cores, self.par_space, tt_train.substituted + [self.index])

    def to_CNF(self):
        return TTExpression(self.value, self.par_space).to_CNF()


class Boolean_Function(Expression):

    def __init__(self, par_space, name: str, tt_e):
        self.name = name
        self.tt_e = tt_e
        super().__init__(par_space, name, [self], lambda x: x)

    def to_tt_train(self):
        return self.tt_e


class Constraint:
    def __init__(self, hypothesis: Hypothesis, tt_expr: TTExpression, hypotheses_to_insert: List[Hypothesis], percent):
        self.tt_expr = tt_expr
        self.hypothesis = hypothesis
        self.hypotheses_to_insert = hypotheses_to_insert
        self.percent = percent

    def get_projection_st(self,  hypothesis: Hypothesis):
        tt_expr = deepcopy(self.tt_expr)
        for h in self.hypotheses_to_insert:
            tt_expr = h.substitute_into(tt_expr)
        bias = tt_leading_entry(tt_expr.cores) - self.percent
        normal = tt_expr.cores[:hypothesis.par_space.atom_count]


class ConstraintSpace(ParameterSpace, ABC):
    def __init__(self):
        """
        self.radius = 1.0
        self.s_lower = 2 ** (-self.dimension) - 1  # -0.9999
        self.projections = []
        self.eq_constraints = []
        self.iq_constraints = [lambda h: jnp.maximum(0, self.s_lower - tt_leading_entry(h))]
        self.faulty_hypothesis = tt_mul_scal(-1, tt_leading_one(self.dimension))
        self.eq_crit = lambda h: sum([jnp.sum(jnp.abs(c(h))) for c in self.eq_constraints])
        self.iq_crit = lambda h: sum([jnp.sum(c(h)) for c in self.iq_constraints])
        self.rank_gradient = D_func(lambda h: 0.0)
        self.boolean_criterion = tt_boolean_criterion(self.dimension)
        """
        self.atom_list: List[Atom] = []
        self.hypothesis_list: List[Hypothesis] = []
        self.iq_constraints: Dict[List] = dict()
        self.eq_constraints: Dict[List] = dict()

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
        a = Atom(name)
        self.atoms.append(a)
        return a

    def Hypothesis(self, name=None):
        h = Hypothesis(self, name)
        self.hypotheses.append(h)
        self.iq_constraints[h] = []
        self.eq_constraints[h] = []
        return h

    def exists_S(self, expr: Expression):
        return self.at_least(expr, -1)

    def at_least(self, expr: Expression, percent):
        tt_train = TTExpression.from_expression(expr)
        relevant_hs = tt_train.hypotheses
        for i, h in enumerate(relevant_hs):
            self.iq_constraints[h].append((tt_train, relevant_hs[:i] + relevant_hs[i+1:], percent))

        def projection(tt_train):
            radius = jnp.sqrt(tt_inner_prod(tt_train, tt_train))
            n = normal_vec(tt_example)
            func_result = tt_inner_prod(tt_train, n)
            if func_result == 0 and offset - 2 * self.s_lower <= 0:
                proj_tt_train = tt_train
                proj_tt_train[0] *= jnp.sqrt(
                    (radius ** 2 - np.abs(0.5 * offset - self.s_lower)) / tt_inner_prod(proj_tt_train, proj_tt_train))
                n = tt_mul_scal((offset - 2 * self.s_lower), n)
                proj_tt_train = tt_add(proj_tt_train, n)
                tt_train = proj_tt_train
            if func_result + offset - 2 * self.s_lower <= 0:
                n = tt_mul_scal(-(func_result / tt_inner_prod(n, n)), n)
                proj_tt_train = tt_add(tt_train, n)
                proj_tt_train[0] *= jnp.sqrt(
                    (radius ** 2 - np.abs(0.5 * offset - self.s_lower)) / tt_inner_prod(proj_tt_train, proj_tt_train))
                n = tt_mul_scal((offset - 2 * self.s_lower) / func_result, n)
                proj_tt_train = tt_add(proj_tt_train, n)
                tt_train = proj_tt_train
            return tt_rank_reduce(tt_train)

        self.projections.append(projection)

    def forall_S(self, example: Expression):
        normal_vec, offset, tt_example = example.to_tt_constraint()
        eq_func = lambda h: jnp.abs(tt_inner_prod(h, normal_vec(tt_example)) + offset - 2)
        self.eq_constraints.append(eq_func)

        def projection(tt_train):
            radius = jnp.sqrt(tt_inner_prod(tt_train, tt_train))
            n = normal_vec(tt_example)
            func_result = tt_inner_prod(tt_train, n)
            if func_result == 0 and np.abs(offset - 2) >= self.s_lower + 1:
                proj_tt_train = tt_train
                proj_tt_train[0] *= jnp.sqrt(
                    (radius ** 2 - np.abs((0.5 * offset - 1))) / tt_inner_prod(proj_tt_train, proj_tt_train))
                n = tt_mul_scal((offset - 2), n)
                proj_tt_train = tt_add(proj_tt_train, n)
                tt_train = proj_tt_train
            elif np.abs(func_result + offset - 2) >= self.s_lower + 1:
                n = tt_mul_scal(-(func_result / tt_inner_prod(n, n)), n)
                proj_tt_train = tt_add(tt_train, n)
                proj_tt_train[0] *= jnp.sqrt(
                    (radius ** 2 - np.abs((0.5 * offset - 1))) / tt_inner_prod(proj_tt_train, proj_tt_train))
                n = tt_mul_scal((offset - 2) / func_result, n)
                proj_tt_train = tt_add(proj_tt_train, n)
                tt_train = proj_tt_train
            return tt_rank_reduce(tt_train)

        self.projections.append(projection)

    def project(self, tt_train):
        criterion_score = self.eq_crit(tt_train) + self.iq_crit(tt_train)
        proj_tt_train = tt_train
        while criterion_score >= self.s_lower + 1:
            for proj in self.projections:
                proj_tt_train = proj(proj_tt_train)
            criterion_score = self.eq_crit(proj_tt_train) + self.iq_crit(proj_tt_train)
        return proj_tt_train

    def normalise(self, tt_train, idx=0):
        return tt_mul_scal(self.radius / np.sqrt(tt_inner_prod(tt_train, tt_train)), tt_train, idx)

    def stopping_criterion(self, tt_train, prev_tt_train):
        return 1 - tt_inner_prod(tt_train, prev_tt_train)

    def round(self, tt_train):
        tt_table = tt_bool_op(tt_train)
        tt_table_p3 = tt_mul_scal(-0.5, tt_hadamard(tt_hadamard(tt_table, tt_table), tt_table))
        tt_table = tt_mul_scal(0.5, tt_table)
        tt_table = tt_rl_orthogonalize(tt_add(tt_table, tt_table_p3))
        tt_update = tt_bool_op_inv(tt_table)
        tt_train = tt_mul_scal(1 - tt_inner_prod(tt_update, tt_train), tt_train)
        tt_train = tt_add(tt_train, tt_update)
        tt_train = tt_mul_scal(1 / jnp.sqrt(tt_inner_prod(tt_train, tt_train)), tt_train)
        return tt_train
