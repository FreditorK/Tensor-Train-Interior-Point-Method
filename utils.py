import random

from sympy.logic.boolalg import ANFform, to_cnf, to_dnf, to_anf
from sympy import symbols
from tt_op import *
from copy import deepcopy
from operators import partial_D, D_func


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
        self.vocab_size = vocab_size
        Atom.counter += 1

    def to_tt_train(self):
        return tt_atom_train(self.index, self.vocab_size)


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

    def __init__(self, name: str, normal_vec, offset, tt_example):
        if name is None:
            self.name = f"e_{str(Meta_Boolean_Function.count)}"
            Meta_Boolean_Function.count += 1
        else:
            self.name = name

        self.normal_vec = normal_vec
        self.offset = offset
        self.tt_example = tt_example

    def to_tt_constraint(self):
        return self.normal_vec, self.offset, self.tt_example

    def __repr__(self):
        return str(self)

    def __str__(self):
        return self.name

    def __or__(self, other):
        e = self.normal_vec
        if isinstance(other, Boolean_Function):
            e = other.normal_vec
        bot = tt_leading_one(len(e))
        bot[0] *= -1
        return Meta_Boolean_Function(
            f"({self.name} v {other.name})",
            lambda e: tt_rl_orthogonalize(tt_add(e, bot)),
            1 + tt_leading_entry(e),
            e
        )

    def __ror__(self, other):
        other.__or__(self)
        return other

    def __xor__(self, other):
        e = self.normal_vec
        if isinstance(other, Boolean_Function):
            e = other.normal_vec
        return Meta_Boolean_Function(
            f"({self.name} ⊻ {other.name})",
            lambda e: tt_neg(e),
            0,
            e
        )

    def __rxor__(self, other):
        return other.__or__(self)

    def __lshift__(self, other):  # <-
        e = self.normal_vec
        if isinstance(other, Boolean_Function):
            e = other.normal_vec
        top = tt_leading_one(len(e))
        return Meta_Boolean_Function(
            f"({self.name} <- {other.name})",
            lambda e: tt_rl_orthogonalize(tt_add(e, top)),
            1 - tt_leading_entry(e),
            e
        )

    def __rlshift__(self, other):
        return other.__lshift(self)

    def __rshift__(self, other):
        e = self.normal_vec
        if isinstance(other, Boolean_Function):
            e = other.normal_vec
        bot = tt_leading_one(len(e))
        bot[0] *= -1
        return Meta_Boolean_Function(
            f"({self.name} <- {other.name})",
            lambda e: tt_rl_orthogonalize(tt_add(e, bot)),
            1 + tt_leading_entry(e),
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


class Boolean_Data(Expression):
    def __init__(self, dataset, labels):
        super().__init__("Dataset", [self], lambda x: x)
        # TODO: Estimate noise level through number of contradicting examples
        self.dataset, indices = np.unique(dataset, return_index=True, axis=0)
        self.labels = labels[indices]
        self.compressed_data = tt_mul_scal(-1, tt_leading_one(self.dataset.shape[1]))
        self._compress()

    def _compress(self):
        # TODO: Might want to consider a divide and conquer here
        for instance, label in zip(self.dataset, self.labels):
            instance_func = bool_to_tt_train(instance)
            if label > 0:
                self.compressed_data = tt_or(self.compressed_data, instance_func)
            else:
                self.compressed_data = tt_and(self.compressed_data, tt_neg(instance_func))

    def to_tt_train(self):
        return deepcopy(self.compressed_data)


class ConstraintSpace:
    def __init__(self, dimension):
        self.dimension = dimension
        self.s_lower = 2 ** (-self.dimension) - 1  # -0.9999
        self.projections = [self._false_projection]
        self.eq_constraints = []
        self.iq_constraints = [lambda h, q=1: jnp.minimum(0, tt_leading_entry(h) - q*self.s_lower) ** 2]
        self.rank_gradient = D_func(lambda h: tt_rank_loss(h))
        self.tt_minus_three = tt_one(dimension)
        self.tt_minus_three[0] *= 3

    def _false_projection(self, tt_train, q=1):
        func_result = tt_leading_entry(tt_train)
        if func_result - q*self.s_lower <= 0:
            one = tt_leading_one(self.dimension)
            tt_train = one
        return tt_train

    def round(self, tt_train, params):
        tt_table = tt_bool_op(tt_train)
        tt_table_p3 = tt_hadamard(tt_hadamard(tt_table, tt_table), tt_table)
        tt_table_p3[0] *= -0.5
        tt_table[0] *= 0.5
        tt_table = tt_rl_orthogonalize(tt_add(tt_table, tt_table_p3))
        tt_update = tt_bool_op_inv(tt_table)
        tt_train[0] *= (1 - tt_inner_prod(tt_update, tt_train))
        tt_train = tt_add(tt_update, tt_train)  # TODO: project this update onto the hyperlane subspace
        tt_train[0] = tt_train[0] / jnp.sqrt(tt_inner_prod(tt_train, tt_train))
        return tt_rank_reduce(tt_train, tt_bound=0)

    def exists_S(self, example: Meta_Boolean_Function):
        normal_vec, offset, tt_example = example.to_tt_constraint()  # TODO: We can pull the sum into the inner product, i.e. add all examples up before?
        iq_func = lambda h, q=1: jnp.minimum(0, tt_inner_prod(h, normal_vec(tt_example)) + offset - q*self.s_lower) ** 2
        self.iq_constraints.append(iq_func)

        def projection(tt_train, q=1):
            tt_train[0] = tt_train[0] / jnp.sqrt(tt_inner_prod(tt_train, tt_train))
            n = normal_vec(tt_example)
            func_result = tt_inner_prod(tt_train, n)
            if func_result + offset - 2*q*self.s_lower <= 0:
                n[0] *= -(1 / tt_inner_prod(n, n)) * func_result
                proj_tt_train = tt_add(tt_train, n)
                n[0] *= (offset - 2*q*self.s_lower)/func_result
                proj_tt_train[0] *= jnp.sqrt((1 - tt_inner_prod(n, n)) / tt_inner_prod(proj_tt_train, proj_tt_train))
                proj_tt_train = tt_add(proj_tt_train, n)
                tt_train = proj_tt_train
            return tt_rl_orthogonalize(tt_train)

        self.projections.append(projection)

    def forall_S(self, example: Meta_Boolean_Function):
        normal_vec, offset, tt_example = example.to_tt_constraint()
        eq_func = lambda h, q=1: (tt_inner_prod(h, normal_vec(tt_example)) + offset - 2*q) ** 2
        self.eq_constraints.append(eq_func)

        def projection(tt_train, q=1):
            tt_train[0] = tt_train[0] / jnp.sqrt(tt_inner_prod(tt_train, tt_train))
            n = normal_vec(tt_example)
            func_result = tt_inner_prod(tt_train, n)
            if np.abs(func_result + offset - q) >= self.s_lower + 1:
                n[0] *= -(1 / tt_inner_prod(n, n)) * func_result
                proj_tt_train = tt_add(tt_train, n)
                n[0] *= (offset - 2*q)/func_result
                proj_tt_train[0] *= jnp.sqrt((1-tt_inner_prod(n, n))/tt_inner_prod(proj_tt_train, proj_tt_train))
                proj_tt_train = tt_add(proj_tt_train, n)
                tt_train = proj_tt_train
            return tt_rl_orthogonalize(tt_train)

        self.projections.append(projection)

    def project(self, tt_train):
        random.shuffle(self.projections)
        proj_tt_train = tt_train
        for proj in self.projections:
            #print(tt_inner_prod(proj_tt_train, proj_tt_train))
            proj_tt_train = proj(proj_tt_train)
        #print(tt_inner_prod(proj_tt_train, proj_tt_train), tt_inner_prod(tt_train, proj_tt_train))
        #print(tt_to_tensor(tt_bool_op(proj_tt_train)))
        #if tt_inner_prod(proj_tt_train, proj_tt_train) < tt_inner_prod(tt_train, proj_tt_train):
         #   print("Knowledge is contradictory. Adjusting expected truth value! ")
        return proj_tt_train


class NoisyConstraintSpace(ConstraintSpace):
    def __init__(self):
        super().__init__()
        self.lr = 1e-1
        self.noise_op_measure = None
        self.noise_gradient = None
        self.iq_gradient = None
        self.expected_truth = 1.0
        self.lr = 1e-2

    def project(self, tt_train):  # TODO: check that projection was contractive!!
        proj_tt_train = tt_train
        for proj in self.projections:
            proj_tt_train = proj(proj_tt_train, self.expected_truth)
        proj_tt_train = tt_rank_reduce(proj_tt_train)
        # Check whether contractive
        if tt_inner_prod(proj_tt_train, proj_tt_train) > tt_inner_prod(tt_train, proj_tt_train):
            print("Knowledge is contradictory. Adjusting expected truth value! ")
            self.expected_truth = max(0, self.expected_truth - self.lr)
        return proj_tt_train
