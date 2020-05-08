from random import choice, random
from equation import ast
import numpy as np

__all__ = ('generate_expression')

PROB_BINARY = 0.7
PROB_UNARY = 0.6
PROB_NEW_VAR = 0.4
PROB_NOT_EQ = 0.3

BINARY_OPS = [ast.OperationExpression.Op.AND, ast.OperationExpression.Op.OR]
UNARY_OPS = [ast.OperationExpression.Op.NOT]


last_variable_index = 1
variables_table = np.array([0], dtype=np.uint8)


def get_variable_name(current_max_index: int, max_index: int):
    global last_variable_index
    global variables_table

    index = 1

    if current_max_index > 1:
        while True:
            if np.sum(variables_table) == current_max_index and current_max_index < max_index:
                current_max_index = current_max_index + 1
                variables_table = np.append(variables_table, [0])

            current_max_index = len(variables_table)
            index = choice([i for i in range(1, current_max_index + 1)])
            variables_table[index - 1] = 1

            if last_variable_index != index:
                last_variable_index = index
                break
    else:
        current_max_index += 1
        variables_table[index - 1] = 1
        variables_table = np.append(variables_table, [0])

    name = 'x{}'.format(index)
    return current_max_index, name


def generate_partial_expression(variable_index: int, max_variables: int, ops_num: int):
    next_variable_index = variable_index

    if ops_num > 0:
        if random() < PROB_BINARY:
            next_variable_index, left = generate_partial_expression(
                next_variable_index, max_variables, int(ops_num / 2))
            next_variable_index, right = generate_partial_expression(
                next_variable_index, max_variables, int(ops_num / 2))

            return next_variable_index, ast.BinaryOperationExpression(choice(BINARY_OPS), left, right)
        elif random() < PROB_UNARY:
            next_variable_index, eq = generate_partial_expression(
                next_variable_index, max_variables, ops_num - 1)
            return next_variable_index, ast.UnaryOperationExpression(choice(UNARY_OPS), eq)

    next_variable_index, name = get_variable_name(
        next_variable_index, max_variables)

    return next_variable_index, ast.Term(name)


def generate_expressions(max_variables: int, ops_num: int):
    global last_variable_index
    global variables_table

    last_variable_index = 1
    variables_table = np.array([0], dtype=np.uint8)

    variable_index = 1

    if ops_num > 1:
        variable_index, left = generate_partial_expression(
            variable_index, max_variables, ops_num / 2)
        variable_index, right = generate_partial_expression(
            variable_index, max_variables, ops_num / 2)
        eq = ast.BinaryOperationExpression(
            choice(BINARY_OPS),
            left,
            right
        )

        if random() < PROB_NOT_EQ:
            eq = ast.UnaryOperationExpression(choice(UNARY_OPS), eq)
    else:
        return

    return ast.Equation(eq)
