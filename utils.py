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
    anf = ANFform(variables, list(reversed(truth_table_labels)))
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

    def __init__(self, name: str, arg, func):
        if name is None:
            self.name = f"e_{str(Meta_Boolean_Function.count)}"
            Meta_Boolean_Function.count += 1
        else:
            self.name = name

        self.arg = arg
        self.func = func

    def to_tt_constraint(self):
        return self.arg, self.func

    def __repr__(self):
        return str(self)

    def __str__(self):
        return self.name

    def __or__(self, other):
        e = self.arg
        if isinstance(other, Boolean_Function):
            e = other.arg
        bot = tt_leading_one(len(e))
        bot[0] *= -1
        return Meta_Boolean_Function(
            f"({self.name} v {other.name})",
            tt_add(e, bot),
            lambda h, idx: 0.5 + 0.5*tt_leading_entry(h) + 0.5*tt_leading_entry(e) - 0.5*tt_inner_prod(h, bond_at(e, idx))
        )

    def __ror__(self, other):
        other.__or__(self)
        return other

    def __xor__(self, other):
        e = self.arg
        if isinstance(other, Boolean_Function):
            e = other.arg
        return Meta_Boolean_Function(
            f"({self.name} ⊻ {other.name})",
            e,
            lambda h, idx: -tt_inner_prod(h, bond_at(e, idx))
        )

    def __rxor__(self, other):
        return other.__or__(self)

    def __lshift__(self, other):  # <-
        e = self.arg
        if isinstance(other, Boolean_Function):
            e = other.arg
        top = tt_leading_one(len(e))
        return Meta_Boolean_Function(
            f"({self.name} <- {other.name})",
            tt_add(e, top),
            lambda h, idx: 0.5 + 0.5 * tt_leading_entry(h) - 0.5 * tt_leading_entry(e) + 0.5 * tt_inner_prod(h, bond_at(e, idx))
        )

    def __rlshift__(self, other):
        return other.__lshift(self)

    def __invert__(self):
        pass

    def __rshift__(self, other):
        pass

    def __rrshift__(self, other):
        pass


class Hypothesis(Meta_Boolean_Function):
    def __init__(self, name=None):
        self.name = name
        if name is None:
            self.name = "hypothesis"
        super().__init__(name, None, None)


class Boolean_Function(Meta_Boolean_Function):
    count = 0

    def __init__(self, expr: Expression, name: str = None):
        if name is None:
            self.name = f"e_{str(Boolean_Function.count)}"
            Boolean_Function.count += 1
        else:
            self.name = name
        tt_e = expr.to_tt_train()
        super().__init__(name, tt_e, lambda x: tt_inner_prod(tt_e, x))


class ConstraintSpace:
    def __init__(self):
        self.projections = []
        self.eq_constraints = []
        self.iq_constraints = []

    def exists_S(self, example: Meta_Boolean_Function):
        e, func = example.to_tt_constraint()

        def penalty(idx):
            return lambda h: func(h, idx) + 1

        self.iq_constraints.append(penalty)

    def forall_S(self, example: Meta_Boolean_Function):
        e, func = example.to_tt_constraint()
        self.eq_constraints.append(lambda h: func(h, -1)-1)
        minus_one = tt_leading_one(len(e))
        minus_one[0] *= -1
        norm = (1/tt_inner_prod(e, e))

        def projection(tt_train):
            ex_0 = deepcopy(e[0])
            tt_train_t = tt_add(tt_train, minus_one)
            e[0] *= -norm*(tt_inner_prod(e, tt_train_t))
            proj = tt_add(tt_train, e)
            e[0] = ex_0
            return tt_rl_orthogonalize(proj)

        self.projections.append(projection)

    def project(self, tt_train):
        for proj in self.projections:
            tt_train = proj(tt_train)
        return tt_train
