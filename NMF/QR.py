import numpy as np
from SVD import Graph, load


def schmidt(A):
    """
    施密特正交化
    :param A:
    :return:
    """
    print(type(A))
    m = A.shape[0]
    n = A.shape[1]
    # 应该是按列计算的，但是访问行比较方便
    A = A.T
    R = np.zeros((n, n))
    # A的第一列
    R[0, 0] = r = np.linalg.norm(A[0])
    q = np.multiply((1 / r), A[0])
    # 存储q
    ql = []
    ql.append(q)
    # A不是第一列
    for i in range(1, n):
        p = np.zeros(m)
        for j in range(0, m):
            if i != j:
                # todo：使用array可以，mat就不可以。结论还是使用array
                # r = np.dot(A[i], ql[j][0,:].T)
                r = np.dot(A[i], ql[j])
                p = np.add(p, np.multiply(r, ql[j]))
                R[j, i] = r
            else:
                r = np.linalg.norm(np.subtract(A[i], p))
                R[j, i] = r
                break
        ql.append(np.multiply((1 / r), np.subtract(A[i], p)))
    Q = np.mat(ql)
    return R, Q.T


if __name__ == '__main__':
    G = load(r"D:\work\learning\NMF\datasets\tu")
    A = G.transferMatrix()
    B = np.mat([[1, -1, 4],
                [1, 4, -2],
                [1, 4, 2],
                [1, -1, 0]])
    Q1, R1 = np.linalg.qr(B)

    R, Q = schmidt(B)
    print("---------------------------")
    print(R)
    print(R1)
    print("------------------------------")
    print(Q)
    print(Q1)
    D = np.dot(Q, R)
    D1 = np.dot(Q1, R1)
    print("__________________________")
    print(D)
    print(D1)
