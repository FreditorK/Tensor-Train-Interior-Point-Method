from utils import *

const_space = ConstraintSpace()
atoms = const_space.generate_atoms(4)
h_0 = const_space.Hypothesis()
h_1 = const_space.Hypothesis()
expression_train = TTExpression.from_expression(atoms[0] & ~h_0 & h_1)
print("Pre-Substitution: ", expression_train.to_CNF())
h_0_expression = atoms[1] | atoms[2]
h_1_expression = atoms[0] ^ atoms[3]
print(f"To substitute in: h_0: {h_0_expression}, h_1: {h_1_expression}")
h_0.value = TTExpression.from_expression(h_0_expression).cores
h_1.value = TTExpression.from_expression(h_1_expression).cores
substituted_expression = h_0.substitute_into(expression_train)
substituted_expression = h_1.substitute_into(substituted_expression)
print("Post-Subsitution:", substituted_expression.to_CNF())
solution = TTExpression.from_expression(atoms[0] & h_1_expression & ~h_0_expression)
print("Truth", solution.to_CNF())