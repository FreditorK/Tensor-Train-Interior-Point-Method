import copy
import random

from sympy.logic.boolalg import ANFform, to_cnf, to_dnf, to_anf
from sympy import symbols
from tt_op import *
from copy import deepcopy
from operators import partial_D, D_func


def _tt_and(x, y):
    print("hi")
    if x is None:
        print("hi_1")
        e = y
        normal_vector = tt_add(e, tt_leading_one(len(e)))
        return tt_leading_entry(e) - 1, normal_vector
    elif y is None:
        print("hi_1")
        e = x
        normal_vector = tt_add(e, tt_leading_one(len(e)))
        return tt_leading_entry(e) - 1, normal_vector
    print(x, y)
    print([c.shape for c in x], [c.shape for c in y])
    return tt_and(x, y)


def _tt_or(x, y):
    if x is None:
        e = y
        normal_vector = tt_add(tt_neg(e), tt_leading_one(len(e)))
        return 1 + tt_leading_entry(e), normal_vector
    elif y is None:
        e = x
        normal_vector = tt_add(tt_neg(e), tt_leading_one(len(e)))
        return 1 + tt_leading_entry(e), normal_vector
    return tt_or(x, y)


def _tt_xor(x, y):
    if x is None:
        e = y
        normal_vector = tt_neg(e)
        return 0, normal_vector
    elif y is None:
        e = x
        normal_vector = tt_neg(e)
        return 0, normal_vector
    return tt_xor(x, y)


def _tt_neg(x):
    if x is None:
        return x  # It only flips the normal vector, i.e. it doesn't matter
    return tt_neg(x)


class Expression:

    def __init__(self, name: str, args, op):
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
            return Boolean_Function(f"({self.name} ∧ {other.name})", tt_and(self.to_tt_train(), other.to_tt_train()))
        return Expression(f"({self.name} ∧ {other.name})", [self, other], _tt_and)

    def __rand__(self, other):
        return other.__and__(self)

    def __or__(self, other):
        if isinstance(self, Atom) and isinstance(other, Atom):
            return Boolean_Function(f"({self.name} v {other.name})", tt_or(self.to_tt_train(), other.to_tt_train()))
        return Expression(f"({self.name} v {other.name})", [self, other], _tt_or)

    def __ror__(self, other):
        return other.__or__(self)

    def __xor__(self, other):
        if isinstance(self, Atom) and isinstance(other, Atom):
            return Boolean_Function(f"({self.name} ⊻ {other.name})", tt_xor(self.to_tt_train(), other.to_tt_train()))
        return Expression(f"({self.name} ⊻ {other.name})", [self, other], _tt_xor)

    def __rxor__(self, other):
        return other.__or__(self)

    def __invert__(self):
        if isinstance(self, Atom):
            return Boolean_Function(f"¬{self.name}", tt_neg(self.to_tt_train()))
        return Expression(f"¬{self.name}", [self], _tt_neg)

    def __lshift__(self, other):  # <-
        return self.__ror__(other.__invert__())

    def __rlshift__(self, other):
        return other.__ror__(self.__invert__())

    def __rshift__(self, other):  # ->
        return other.__ror__(self.__invert__())

    def __rrshift__(self, other):  # ->
        return self.__ror__(other.__invert__())


class Atom(Expression):
    count = 0

    def __init__(self, name=None):
        if name is None:
            name = f"a_{str(Atom.count)}"
        super().__init__(name, [self], lambda x: x)
        self.index = Atom.count
        Atom.count += 1

    def to_tt_train(self):
        return tt_atom_train(self.index, Atom.count)


class Hypothesis(Expression):
    count = 0

    def __init__(self, name=None, indices=None):
        if name is None:
            name = f"h_{Hypothesis.count}"
        super().__init__(name, [self], lambda x: x)
        self.active = True
        self.indices = indices
        if indices is None:
            self.tt_hypothesis = tt_leading_one(Atom.count)
        else:
            self.tt_hypothesis = tt_leading_one(len(self.indices))
        Hypothesis.count += 1

    def to_tt_train(self):
        if self.active:
            if self.indices is None:
                return self.tt_hypothesis
            tt_train = [np.array([1, 0]).reshape(1, 2, 1) for _ in range(Atom.count)]
            for idx, i in enumerate(self.indices):
                tt_train[i] = self.tt_hypothesis[idx]
            return tt_train
        return None


class Boolean_Function(Expression):

    def __init__(self, name: str, tt_e):
        self.name = name
        self.tt_e = tt_e
        super().__init__(name, [self], lambda x: x)

    def to_tt_train(self):
        return self.tt_e


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


def generate_atoms(n):
    atoms = []
    for i in range(n):
        atoms.append(Atom(f"x_{i}"))
    return atoms


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


class ConstraintSpace:
    def __init__(self, dimension):
        self.dimension = dimension
        self.radius = 1.0
        self.s_lower = 2 ** (-self.dimension) - 1  # -0.9999
        self.projections = []
        self.eq_constraints = []
        self.iq_constraints = [lambda h: jnp.maximum(0, self.s_lower - tt_leading_entry(h))]
        self.faulty_hypothesis = tt_mul_scal(-1, tt_leading_one(dimension))
        self.eq_crit = lambda h: sum([jnp.sum(jnp.abs(c(h))) for c in self.eq_constraints])
        self.iq_crit = lambda h: sum([jnp.sum(c(h)) for c in self.iq_constraints])
        self.rank_gradient = D_func(lambda h: 0.0)
        self.boolean_criterion = tt_boolean_criterion(dimension)

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

    def exists_S(self, example: Expression):
        normal_vec, offset, tt_example = example.to_tt_constraint()  # TODO: We can pull the sum into the inner product, i.e. add all examples up before?
        iq_func = lambda h: jnp.maximum(0, self.s_lower - tt_inner_prod(h, normal_vec(tt_example)) - offset)
        self.iq_constraints.append(iq_func)

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


class NoisyConstraintSpace(ConstraintSpace):
    def __init__(self, dimension):
        super().__init__(dimension)
        self.dimension = dimension
        self.radius = 1.0
        self.s_lower = 2 ** (-self.dimension) - 1  # -0.9999
        self.projections = [self._false_projection]
        self.eq_constraints = []
        self.iq_constraints = [lambda h: jnp.maximum(0, self.s_lower - tt_leading_entry(h))]
        self.faulty_hypothesis = tt_mul_scal(-1, tt_leading_one(dimension))
        self.eq_crit = lambda h: sum([jnp.sum(jnp.abs(c(h))) for c in self.eq_constraints])
        self.iq_crit = lambda h: sum([jnp.sum(c(h)) for c in self.iq_constraints])
        self.rank_gradient = D_func(lambda h: tt_inf_schatten_norm(h))

        self.noise_op_measure = np.ones(dimension)
        self.noise_gradient = partial_D(tt_noise_loss, 1)
        self.denoise_gradient = partial_D(tt_denoise_loss, 1)
        self.noisy_boolean_criterion = tt_noisy_boolean_criterion(dimension)

    def boolean_criterion(self, tt_train):
        return self.noisy_criterion(tt_train, self.noise_op_measure)

    def _false_projection(self, tt_train, q=1):
        func_result = tt_leading_entry(tt_train)
        if func_result - q * self.s_lower <= 0:
            one = tt_leading_one(self.dimension)
            tt_train = tt_mul_scal(np.sqrt(q), one)
        return tt_train

    def round(self, tt_train):
        tt_table = tt_bool_op(tt_train)
        tt_table_p3 = tt_mul_scal(-0.5, tt_hadamard(tt_hadamard(tt_table, tt_table), tt_table))
        tt_table = tt_mul_scal(0.5, tt_table)
        tt_table = tt_rl_orthogonalize(tt_add(tt_table, tt_table_p3))
        tt_update = tt_bool_op_inv(tt_table)
        tt_train = tt_mul_scal(1 - tt_inner_prod(tt_update, tt_train), tt_train)
        tt_train = tt_add(tt_train, tt_update)  # TODO: project this update onto the hyperlane subspace
        tt_train = tt_mul_scal(1 / jnp.sqrt(tt_inner_prod(tt_train, tt_train)), tt_train)
        return tt_rank_reduce(tt_train, tt_bound=0)

    def exists_S(self, example: Expression):
        normal_vec, offset, tt_example = example.to_tt_constraint()  # TODO: We can pull the sum into the inner product, i.e. add all examples up before?
        iq_func = lambda h, q=1: jnp.maximum(0, q * self.s_lower - tt_inner_prod(h, normal_vec(tt_example)) - offset)
        self.iq_constraints.append(iq_func)

        def projection(tt_train, q=1):
            radius = jnp.sqrt(tt_inner_prod(tt_train, tt_train))
            n = normal_vec(tt_example)
            func_result = tt_inner_prod(tt_train, n)
            if func_result + offset - 2 * q * self.s_lower <= 0:
                n = tt_mul_scal(-(func_result / tt_inner_prod(n, n)), n)
                proj_tt_train = tt_add(tt_train, n)
                proj_tt_train[0] *= jnp.sqrt(
                    (radius ** 2 - np.abs(0.5 * offset - q * self.s_lower)) / tt_inner_prod(proj_tt_train,
                                                                                            proj_tt_train))
                n = tt_mul_scal((offset - 2 * q * self.s_lower) / func_result, n)
                proj_tt_train = tt_add(proj_tt_train, n)
                tt_train = proj_tt_train
            return tt_rl_orthogonalize(tt_train)

        self.projections.append(projection)

    def forall_S(self, example: Expression):
        normal_vec, offset, tt_example = example.to_tt_constraint()
        eq_func = lambda h, q=1: jnp.abs(tt_inner_prod(h, normal_vec(tt_example)) + offset - 2 * q)
        self.eq_constraints.append(eq_func)

        def projection(tt_train, q=1):
            radius = jnp.sqrt(tt_inner_prod(tt_train, tt_train))
            n = normal_vec(tt_example)
            func_result = tt_inner_prod(tt_train, n)
            if np.abs(func_result + offset - 2 * q) >= self.s_lower + 1:
                n = tt_mul_scal(-(func_result / tt_inner_prod(n, n)), n)
                proj_tt_train = tt_add(tt_train, n)
                proj_tt_train[0] *= jnp.sqrt(
                    (radius ** 2 - np.abs((0.5 * offset - q))) / tt_inner_prod(proj_tt_train, proj_tt_train))
                n = tt_mul_scal((offset - 2 * q) / func_result, n)
                proj_tt_train = tt_add(proj_tt_train, n)
                tt_train = proj_tt_train
            return tt_rl_orthogonalize(tt_train)

        self.projections.append(projection)

    def project(self, tt_train):
        criterion_score = self.eq_crit(tt_train) + self.iq_crit(tt_train)
        proj_tt_train = tt_train
        while criterion_score > self.s_lower + 1:
            for proj in self.projections:
                proj_tt_train = proj(proj_tt_train)
            criterion_score = self.eq_crit(proj_tt_train) + self.iq_crit(proj_tt_train)
        return tt_rank_reduce(proj_tt_train)
