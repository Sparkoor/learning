import numpy as np
from collections import defaultdict


class Graph(defaultdict):
    def __init__(self):
        super(Graph, self).__init__(list)

    def countNode(self):
        return len(self.keys())

    def transferMatrix(self):
        m = self.countNode()
        A = np.zeros((m, m))
        for node, adj in self.items():
            for a in adj:
                A[node - 1, a - 1] = 1
        return A


def load(filePath):
    graph = Graph()
    with open(filePath) as f:
        for i in f:
            x, y = i.split(",")[:2]
            x = int(x)
            y = int(y)
            graph[x].append(y)
    return graph


if __name__ == '__main__':
    G = load(r"D:\work\learning\NMF\datasets\tu")
    print(G)
    A = G.transferMatrix()
    print(A)
    U, sigm, VT = np.linalg.svd(A)
    print(U)
    print(sigm)
    print(VT)
