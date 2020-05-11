from uuid import uuid4
from graphviz import Graph
from abc import ABC, abstractmethod


class Visitor(ABC):
    def visit(self, node: 'Node'):
        return node.accept(self)

    @abstractmethod
    def visit_equation(self, node: 'Equation'):
        pass

    @abstractmethod
    def visit_unary_op(self, node: 'UnaryOperationExpression'):
        pass

    @abstractmethod
    def visit_binary_op(self, node: 'BinaryOperationExpression'):
        pass

    @abstractmethod
    def visit_term(self, node: 'Term'):
        pass


###############################################################################
# AST
###############################################################################


class Node(ABC):
    class Kind:
        TERM = 1
        OP_EXPR = 2
        EQUATION = 3

    @abstractmethod
    def kind(self):
        pass

    @abstractmethod
    def accept(self, visitor: Visitor):
        pass


class Term(Node):
    def __init__(self, name: str):
        self.name = name

    def kind(self):
        return Node.Kind.TERM

    def accept(self, visitor: Visitor):
        return visitor.visit_term(self)


class Equation(Node):
    def __init__(self, child: Node):
        self.child = child

    def kind(self):
        return Node.Kind.EQUATION

    def accept(self, visitor: Visitor):
        return visitor.visit_equation(self)


class Expression(Node, ABC):
    class Type:
        UNARY = 1
        BINARY = 2


class OperationExpression(Expression, ABC):
    class Op:
        # Standard basis
        AND = 1
        OR = 2
        NOT = 3

    def kind(self):
        return Node.Kind.OP_EXPR

    @abstractmethod
    def type(self):
        pass


class UnaryOperationExpression(OperationExpression):
    def __init__(self, op: int, child: Node):
        self.op = op
        self.child = child

    def type(self):
        return Expression.Type.UNARY

    def accept(self, visitor: Visitor):
        return visitor.visit_unary_op(self)


class BinaryOperationExpression(OperationExpression):
    def __init__(self, op: int, left: Node, right: Node):
        self.op = op
        self.left = left
        self.right = right

    def type(self):
        return Expression.Type.BINARY

    def accept(self, visitor: Visitor):
        return visitor.visit_binary_op(self)


###############################################################################
# StringVisitor
###############################################################################

OPERATION_SYMBOLS = {
    OperationExpression.Op.AND: '&',
    OperationExpression.Op.NOT: '!',
    OperationExpression.Op.OR: '|'
}


class StringVisitor(Visitor):
    def visit_equation(self, node: Equation):
        return node.child.accept(self)

    def visit_unary_op(self, node: UnaryOperationExpression) -> str:
        child = ('({0})' if isinstance(node.child,
                                       BinaryOperationExpression) else '{0}').format(
            node.child.accept(self))

        return '{1}{0}'.format(
            child,
            OPERATION_SYMBOLS[node.op]
        )

    def visit_binary_op(self, node: BinaryOperationExpression) -> str:
        left = ('({0})' if isinstance(node.left,
                                      BinaryOperationExpression) else '{0}').format(
            node.left.accept(self))
        right = ('({0})' if isinstance(node.right,
                                       BinaryOperationExpression) else '{0}').format(
            node.right.accept(self))

        return '{0}{2}{1}'.format(
            left,
            right,
            OPERATION_SYMBOLS[node.op]
        )

    def visit_term(self, node: Term):
        return '{0}'.format(node.name)


###############################################################################
# GraphvizVisitor
###############################################################################


class GraphvizVisitor(Visitor):
    def __init__(self):
        self._graph = Graph('equation')
        self._subgraph_terms = Graph('terms')
        self._terms = {}

        self._graph.attr(rankdir='BT', ordering='out') #, splines='false')
        self._subgraph_terms.attr(rank='same', rankdir='LR')

    def visit(self, node: 'Node') -> Graph:
        node.accept(self)

        self._graph.subgraph(self._subgraph_terms)

        return self._graph

    def visit_equation(self, node: Equation):
        uid = str(uuid4())
        child_uid = node.child.accept(self)

        self._graph.node(uid, label='F', shape='square')
        self._graph.edge(uid, child_uid, penwidth='3')

        return uid

    def visit_unary_op(self, node: UnaryOperationExpression) -> str:
        uid = str(uuid4())
        child_uid = node.child.accept(self)

        self._graph.node(uid, label=OPERATION_SYMBOLS[node.op], shape='invtriangle')
        self._graph.edge(uid, child_uid, penwidth='3')

        return uid

    def visit_binary_op(self, node: BinaryOperationExpression) -> str:
        uid = str(uuid4())
        left_uid = node.left.accept(self)
        right_uid = node.right.accept(self)

        self._graph.node(uid, label=OPERATION_SYMBOLS[node.op], shape='invtriangle')
        self._graph.edge(uid, left_uid, penwidth='3')
        self._graph.edge(uid, right_uid, penwidth='3')

        return uid

    def visit_term(self, node: Term):
        name = node.name
        uid = self._terms[name] if name in self._terms else str(uuid4())
        self._terms[node.name] = uid

        self._subgraph_terms.node(uid, label=name, shape='circle') # shape='point')

        return uid


###############################################################################
# Test
###############################################################################

if __name__ == '__main__':
    def test():
        string_visitor = StringVisitor()
        graphviz_visitor = GraphvizVisitor()

        data = Equation(
            BinaryOperationExpression(
                OperationExpression.Op.OR,
                BinaryOperationExpression(
                    OperationExpression.Op.OR,
                    BinaryOperationExpression(
                        OperationExpression.Op.AND,
                        Term('x1'),
                        Term('x2'),
                    ),
                    BinaryOperationExpression(
                        OperationExpression.Op.AND,
                        Term('x1'),
                        Term('x3'),
                    )
                ),
                BinaryOperationExpression(
                    OperationExpression.Op.AND,
                    Term('x2'),
                    Term('x3'),
                )
            )
        )

        # a^(!a|b)
        result = string_visitor.visit(data)
        graph = graphviz_visitor.visit(data)

        graph.format = 'PNG'
        graph.render()

    test()
