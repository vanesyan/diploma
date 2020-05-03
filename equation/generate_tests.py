from shutil import rmtree
import os
from os import path
from random import randint

from generate_expression import generate_expressions
from ast import StringVisitor, GraphvizVisitor

def generate(n: int):
    if path.exists('./data'):
        rmtree('./data')        
    os.mkdir('./data')

    for i in range(1, n+1):
        string_visitor = StringVisitor()
        graphviz_visitor = GraphvizVisitor()

        vars_num = randint(1, 10)
        ops_num = vars_num + randint(0, 4)
        eq = generate_expressions(vars_num, ops_num)
        dirname = '{}_{}_{}'.format(i, vars_num, ops_num)
        dirpath = './data/{}'.format(dirname)

        os.mkdir(dirpath)

        with open('{}/result.txt'.format(dirpath), 'w+') as file:
            string_repr = string_visitor.visit(eq)
            file.write(string_repr)

        graph_repr = graphviz_visitor.visit(eq)
        graph_repr.format = 'PNG'
        graph_repr.render(filename='{}/image'.format(dirpath))

generate(4)

