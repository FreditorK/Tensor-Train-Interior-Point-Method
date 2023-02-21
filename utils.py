from sympy.logic.boolalg import ANFform, to_cnf, to_dnf, to_anf
from sympy import symbols
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


def get_ANF(atoms, hypothesis):
    variable_names = " ".join([a.name for a in atoms])
    variables = symbols(variable_names)
    truth_table_labels = ((np.round(tt_to_tensor(tt_bool_op(hypothesis))) + 1) / 2).astype(int).flatten()
    anf = ANFform(variables, truth_table_labels)
    return anf


def get_CNF(atoms, hypothesis):
    anf = get_ANF(atoms, hypothesis)
    return to_cnf(anf, simplify=True)


def get_DNF(atoms, hypothesis):
    anf = get_ANF(atoms, hypothesis)
    return to_dnf(anf, simplify=True)


def influence_geq(atom, eps):
    def influence_constraint(_):
        idx = atom.index
        return lambda h: tt_influence(h, idx) - eps

    return influence_constraint


def influence_leq(atom, eps):
    def influence_constraint(_):
        idx = atom.index
        return lambda h: eps - tt_influence(h, idx)

    return influence_constraint


class Meta_Boolean_Function:
    count = 0

    def __init__(self, name: str, args, bias, inner_prod_sgn):
        if name is None:
            self.name = f"e_{str(Meta_Boolean_Function.count)}"
            Meta_Boolean_Function.count += 1
        else:
            self.name = name

        self.bias = bias
        self.inner_prod_sgn = inner_prod_sgn
        self.args = args

    def to_tt_constraint(self, negation=False):
        example = next(func for func in self.args if not isinstance(func, Hypothesis))
        e, _, _ = example.to_tt_constraint()
        if negation:
            return e, self.bias(e) + 2, self.inner_prod_sgn
        return e, self.bias(e) - 2, self.inner_prod_sgn

    def __repr__(self):
        return str(self)

    def __str__(self):
        return self.name

    def __and__(self, other):
        return Meta_Boolean_Function(f"({self.name} ∧ {other.name})", [self, other], lambda e: -1 + tt_leading_entry(e),
                                     1)

    def __rand__(self, other):
        return other.__and__(self)

    def __or__(self, other):
        return Meta_Boolean_Function(f"({self.name} v {other.name})", [self, other], lambda e: 1 + tt_leading_entry(e),
                                     -1)

    def __ror__(self, other):
        other.__or__(self)
        return other

    def __xor__(self, other):
        return Meta_Boolean_Function(f"({self.name} ⊻ {other.name})", [self, other], lambda e: 0, -1)

    def __rxor__(self, other):
        return other.__or__(self)

    def __lshift__(self, other):  # <-
        return Meta_Boolean_Function(f"({self.name} <- {other.name})", [self, other], lambda e: 1 - tt_leading_entry(e),
                                     1)

    def __rlshift__(self, other):
        return other.__lshift(self)


class Hypothesis(Meta_Boolean_Function):
    def __init__(self, name=None):
        if name is None:
            self.name = "hypothesis"
        self.name = name
        super().__init__(name, [self], lambda x: 0, 0)


class Boolean_Function(Meta_Boolean_Function):
    count = 0

    def __init__(self, expr: Expression, name: str = None):
        if name is None:
            self.name = f"e_{str(Boolean_Function.count)}"
            Boolean_Function.count += 1
        else:
            self.name = name
        super().__init__(name, [expr], lambda x: 0, 0)

    def to_tt_constraint(self, negation=False):
        example = next(expr for expr in self.args if isinstance(expr, Expression))
        e = example.to_tt_train()
        return e, 0, 0


class ConstraintSpace:
    def __init__(self):
        self.forall_constraints = []
        self.not_forall_constraints = []
        self.eq_constraints = []
        self.exists_constraints = []
        self.not_exists_constraints = []

    def exists_S(self, example: Meta_Boolean_Function):
        e, bias, sgn = example.to_tt_constraint(negation=False)

        def penalty(idx):
            e_bonded = e
            if idx != -1:
                e_bonded = bond_at(e, idx)
            return lambda h: -bias + tt_leading_entry(h) + sgn * tt_inner_prod(h, e_bonded)

        self.exists_constraints.append(penalty)

    def not_exists_S(self, example: Meta_Boolean_Function):
        e, bias, sgn = example.to_tt_constraint(negation=True)

        def penalty(idx):
            e_bonded = e
            if idx != -1:
                e_bonded = bond_at(e, idx)
            return lambda h: -bias + tt_leading_entry(h) + sgn * tt_inner_prod(h, e_bonded)

        self.not_exists_constraints.append(penalty)

    def forall_S(self, example: Meta_Boolean_Function):
        e, bias, sgn = example.to_tt_constraint(negation=False)
        plane_eq = lambda h: tt_inner_prod(h, e) + tt_leading_entry(h) -tt_leading_entry(e) - 1
        self.eq_constraints.append(plane_eq)
        minus_one = tt_leading_one(len(e))
        minus_one[0] *= -1
        one = tt_leading_one(len(e))

        def projection(tt_train):
            ex_t = tt_add(e, one)
            tt_train_t = tt_add(tt_train, minus_one)
            ex_t[0] *= -(1/tt_inner_prod(ex_t, ex_t))*(tt_inner_prod(ex_t, tt_train_t))
            proj = tt_add(tt_train, ex_t)
            return proj

        self.forall_constraints.append(projection)

    def not_forall_S(self, example: Meta_Boolean_Function):
        e, bias, sgn = example.to_tt_constraint(negation=False)
        self.eq_constraints.append(lambda h: bias + tt_leading_entry(h) + sgn * tt_inner_prod(h, e))
        one = tt_leading_one(len(e))
        ex = tt_add(e, one)

        def projection(tt_train):
            ex[0] *= -(sgn * tt_inner_prod(ex, tt_train) + bias)
            proj = tt_add(tt_train, ex)
            return tt_rl_orthogonalize(proj)

        self.not_forall_constraints.append(projection)

    def _return_inequality_constraints(self):
        return self.exists_constraints + self.not_exists_constraints

    def project(self, tt_train):
        for proj in self.forall_constraints + self.not_forall_constraints:
            tt_train = proj(tt_train)
        return tt_train
