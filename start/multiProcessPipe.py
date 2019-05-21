"""
使用管道
"""
from multiprocessing import Pipe, Queue, Process
from concurrent.futures import ProcessPoolExecutor
import concurrent
from collections import defaultdict


class Graph(defaultdict):
    def __init__(self):
        super(Graph, self).__init__(list)


def a(s, pip):
    print(type(pip))
    close, input_pipe = pip
    close.colse()
    print(s)
    try:
        item = input_pipe.recv()
        print(item)
    except EOFError:
        print("错误")


def b(pipe):
    g = Graph()
    s = [('yellow', 1), ('blue', 2), ('yellow', 3), ('yellow', 3), ('blue', 4), ('red', 1)]
    for i, j in s:
        g[i].append(j)
    output, _ = pipe
    output.send(g)
    print(pipe)
    with ProcessPoolExecutor(max_workers=2) as executor:
        features = [executor.submit(a, x, output) for x in [1, 2]]
        for fe in concurrent.futures.as_completed(features):
            print(fe.result())


def c():
    pipe = Pipe(True)


if __name__ == '__main__':
    pipe = Pipe(True)
    b(pipe)
