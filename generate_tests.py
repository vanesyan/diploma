from shutil import rmtree
import os
from os import path
from random import randint
import time
import numpy as np

from generate_expression import generate_expressions
from equation import ast
from node_detection import recognize
from diff_images import diff_images
import multiprocessing

TIMEOUT = 30


def generate(n: int):
    if path.exists('./data'):
        rmtree('./data')
    os.mkdir('./data')

    tests = []

    for i in range(1, n+1):
        string_visitor = ast.StringVisitor()
        graphviz_visitor = ast.GraphvizVisitor()

        vars_num = randint(2, 8)
        ops_num = vars_num + randint(0, 4)
        eq = generate_expressions(vars_num, ops_num)
        dirname = '{}_{}_{}'.format(i, vars_num, ops_num)
        dirpath = './data/{}'.format(dirname)

        os.mkdir(dirpath)

        graph_repr = graphviz_visitor.visit(eq)
        graph_repr.format = 'PNG'
        graph_repr.render(filename='{}/image'.format(dirpath))

        tests.append(dirpath)

    return tests


def test_batch(batch_size):
    paths = generate(batch_size)
    passed = 0
    paren_conn, child_conn = multiprocessing.Pipe()

    def proc(path, p):
        passed = True
        text = ''
        try:
            text = recognize('{}/image.png'.format(path), debug=False)
            diff = diff_images('{}/image.png'.format(path),
                               '{}/image_result.png'.format(path), debug=False)

            print('Pixels diff', diff)

            if diff > 0:
                passed = False

            with open('{}/text_result.txt'.format(path), 'w') as f:
                f.write('OK' if passed else 'FAILED')
        except Exception as e:
            # print(e)
            passed = False

        with open('{}/test.txt'.format(path), 'w') as f:
            f.write('OK' if passed else 'FAILED')

        p.send(passed)
        p.close()

    procs = []

    for path in paths:
        p = multiprocessing.Process(
            target=proc, name="proc", args=(path, child_conn,))
        p.start()
        procs.append(p)

    start = time.time()
    bool_list = [True]*batch_size
    while time.time() - start <= TIMEOUT * batch_size:
        for i in range(batch_size):
            bool_list[i] = procs[i].is_alive()

        if np.any(bool_list):
            time.sleep(.1)
        else:
            break
    else:
        print("timed out, killing all processes")
        for p in procs:
            p.terminate()

    for p in procs:
        p.join()

    while paren_conn.poll():
        result = paren_conn.recv()
        if result:
            passed += 1

    print('Total passed: {}'.format(passed))

    return passed


def main():
    # batch_sizes = [10, 50, 100, 150, 200]
    batch_sizes = [5]
    result = []
    avg_result = []

    for index, batch_size in enumerate(batch_sizes):
        print('Round: {}, batch_size: {}, running {} times'.format(index + 1, batch_size, 3))
        print('========================================================')

        subresult = []
        for _ in range(3):
            subresult.append(test_batch(batch_size))

        avg = np.average(np.array(subresult))
        result.append(subresult)
        avg_result.append(avg)

    print(avg_result)
    print(result)


if __name__ == '__main__':
    main()
