from sympy.logic.boolalg import ANFform, to_cnf, to_dnf, to_anf
from sympy import symbols
from tt_op import *
from copy import deepcopy
from operators import partial_D


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
            name = f"a_{str(Atom.counter)}"
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

    def __init__(self, name: str, mod_tt_example, func, tt_example):
        if name is None:
            self.name = f"e_{str(Meta_Boolean_Function.count)}"
            Meta_Boolean_Function.count += 1
        else:
            self.name = name

        self.mod_tt_example = mod_tt_example
        self.func = func
        self.tt_example = tt_example

    def to_tt_constraint(self):
        return self.mod_tt_example, self.func, self.tt_example

    def __repr__(self):
        return str(self)

    def __str__(self):
        return self.name

    def __or__(self, other):
        e = self.mod_tt_example
        if isinstance(other, Boolean_Function):
            e = other.mod_tt_example
        bot = tt_leading_one(len(e))
        bot[0] *= -1
        return Meta_Boolean_Function(
            f"({self.name} v {other.name})",
            lambda e, q=1: tt_rl_orthogonalize(tt_add(e, bot)),
            lambda h, q=1: q * (0.5 + 0.5 * tt_leading_entry(h) + 0.5 * tt_leading_entry(e)) - 0.5 * tt_inner_prod(h,
                                                                                                                   e),
            e
        )

    def __ror__(self, other):
        other.__or__(self)
        return other

    def __xor__(self, other):
        e = self.mod_tt_example
        if isinstance(other, Boolean_Function):
            e = other.mod_tt_example
        return Meta_Boolean_Function(
            f"({self.name} ⊻ {other.name})",
            lambda e, q=1: e,
            lambda h, q=1: -tt_inner_prod(h, e),
            e
        )

    def __rxor__(self, other):
        return other.__or__(self)

    def __lshift__(self, other):  # <-
        e = self.mod_tt_example
        if isinstance(other, Boolean_Function):
            e = other.mod_tt_example
        top = tt_leading_one(len(e))
        return Meta_Boolean_Function(
            f"({self.name} <- {other.name})",
            lambda e, q=1: tt_rl_orthogonalize(tt_add(e, tt_mul_scal(q, top))),
            lambda h, q=1: q * (0.5 + 0.5 * tt_leading_entry(h) - 0.5 * tt_leading_entry(e)) + 0.5 * tt_inner_prod(h,
                                                                                                                   e),
            e
        )

    def __rlshift__(self, other):
        return other.__lshift(self)

    def __rshift__(self, other):
        e = self.mod_tt_example
        if isinstance(other, Boolean_Function):
            e = other.mod_tt_example
        bot = tt_leading_one(len(e))
        bot[0] *= -1
        return Meta_Boolean_Function(
            f"({self.name} <- {other.name})",
            lambda e, q=1: tt_rl_orthogonalize(tt_add(e, bot)),
            lambda h, q=1: q * (0.5 - 0.5 * tt_leading_entry(h) + 0.5 * tt_leading_entry(e)) + 0.5 * tt_inner_prod(h,
                                                                                                                   e),
            e
        )

    def __rrshift__(self, other):
        return other.__rshift(self)


class Hypothesis(Meta_Boolean_Function):
    def __init__(self, name=None):
        self.name = name
        if name is None:
            self.name = "hypothesis"
        super().__init__(name, None, None, None)


class Boolean_Function(Meta_Boolean_Function):
    count = 0

    def __init__(self, expr: Expression, name: str = None):
        if name is None:
            self.name = f"e_{str(Boolean_Function.count)}"
            Boolean_Function.count += 1
        else:
            self.name = name
        tt_e = expr.to_tt_train()
        super().__init__(name, tt_e, lambda x: tt_inner_prod(tt_e, x), tt_e)


class ConstraintSpace:
    def __init__(self):
        self.projections = []
        self.eq_constraints = []
        self.inequalities = []
        self.iq_constraints = [lambda h, q=1: jnp.minimum(0, q * tt_leading_entry(h) + q - 1e-4) ** 2]
        self.iq_gradient = None

    def gradient(self, tt_train):
        if self.iq_gradient is None:
            n = len(self.iq_constraints)
            self.iq_gradient = partial_D(lambda *h: (1/n)*sum([loss(h) for loss in self.iq_constraints]), 0)
        return self.iq_gradient(*tt_train)

    def update_noise_lvl(self, tt_train):
        ...

    def exists_S(self, example: Meta_Boolean_Function):
        _, func, tt_example = example.to_tt_constraint()  # TODO: We can pull the sum into the inner product, i.e. add all examples up before?
        iq_func = lambda h, q=1: jnp.minimum(0, func(h, q) + q - 1e-5) ** 2
        self.iq_constraints.append(iq_func)

    def forall_S(self, example: Meta_Boolean_Function):
        mod_tt_example, func, tt_example = example.to_tt_constraint()
        self.eq_constraints.append(lambda h, q=1: func(h) - q)

        def projection(tt_train, q=1):
            ex_0 = mod_tt_example(tt_example, q)
            norm = (1 / tt_inner_prod(ex_0, ex_0))
            ex_0[0] *= -norm * (2 * func(tt_train) - 2 * q)
            proj = tt_add(tt_train, ex_0)
            return tt_rl_orthogonalize(proj)

        self.projections.append(projection)

    def project(self, tt_train):
        for proj in self.projections:
            tt_train = proj(tt_train)
        return tt_train


class NoisyConstraintSpace(ConstraintSpace):
    def __init__(self):
        super().__init__()
        self.lr = 1e-1
        self.noise_op_measure = None
        self.noise_gradient = None
        self.iq_gradient = None
        self.expected_truth = 1.0
        self.expected_truth_gradient = None

    def gradient(self, tt_train):
        if self.iq_gradient is None:
            self.noise_op_measure = np.ones(len(tt_train))

            def gradient(p, q, *h):
                h = tt_noise_op(h, p)
                return sum([loss(h, q) for loss in self.iq_constraints])

            self.iq_gradient = partial_D(gradient, 2)
        gradient = self.iq_gradient(self.noise_op_measure, self.expected_truth, *tt_train)
        return gradient

    def update_noise_lvl(self, tt_train, iterations=5):
        if self.noise_gradient is None:
            def noise_gradient(p, q, h):
                h = tt_noise_op(h, p)
                return sum([loss(h, q) for loss in self.iq_constraints]) + sum(
                    loss(h, q) ** 2 for loss in self.eq_constraints)

            self.noise_gradient = partial_D(noise_gradient, 0)  # TODO: We can probably add everything up for the proj loss
            self.expected_truth_gradient = partial_D(noise_gradient, 1)
        gradient = self.noise_gradient(self.noise_op_measure, self.expected_truth, tt_train)
        is_zero = self.noise_op_measure*(1-self.noise_op_measure)*(np.abs(gradient) < 1e-5)
        self.noise_op_measure = np.clip(self.noise_op_measure + self.lr * (is_zero - gradient), a_max=1, a_min=0)
        expected_truth_gradient = self.expected_truth_gradient(self.noise_op_measure, self.expected_truth, tt_train)
        is_zero_2 = self.expected_truth*(1-self.expected_truth)*(np.abs(expected_truth_gradient) < 1e-5)
        self.expected_truth = np.clip(self.expected_truth + self.lr *(is_zero_2 - expected_truth_gradient), a_max=1, a_min=0)
        print(self.expected_truth, self.noise_op_measure)
        if iterations > 0:
            self.update_noise_lvl(tt_train, iterations-1)


    def project(self, tt_train): # TODO: check that projection was contractive!!
        noisy_tt_train = tt_noise_op(tt_train, self.noise_op_measure)
        for proj in self.projections:
            noisy_tt_train = proj(noisy_tt_train, self.expected_truth)
        proj_tt_train = tt_rl_orthogonalize(tt_noise_op_inv(noisy_tt_train, self.noise_op_measure))
        return proj_tt_train
