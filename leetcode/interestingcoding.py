"""
编程的乐趣的算法
"""


class Solution:
    def convertToDemical(self, r, d, rep):
        """
        转化成十进制
        :param r:
        :param d:
        :param rep:
        :return:
        """
        number = 0
        for i in range(d - 1):
            # 刚好和i是反的
            number = (number + rep[i]) * r
        number += rep[d - 1]
        return number

    def howHardIsTheCrystal(self, n, d):
        """
         有几个小球，测试从几楼扔球下来会碎，d球的高度搜索
         需要人工判断球碎不碎
        @:param n 楼的层
        @:param d 球的个数
        :return:
        """
        # 进制
        r = 1
        # 计算进制
        while r ** d <= n:
            r = r + 1
        print("radio chosen is ", r)
        # 测试次数
        numDrops = 0
        # 保存d位
        floorNoBlack = [0] * d
        for i in range(d):
            for j in range(r - 1):
                floorNoBlack[i] += 1
                Floor = self.convertToDemical(r, d, floorNoBlack)
                if Floor > n:
                    floorNoBlack[i] -= 1
                    break
                print('drop ball', i + 1, 'from floor', Floor)
                yes = input('Did the ball break(yes or no)?:')
                numDrops += 1
                if yes == 'yes':
                    floorNoBlack[i] -= 1
                    break
        hardness = self.convertToDemical(r, d, floorNoBlack)
        return hardness, numDrops


if __name__ == '__main__':
    S = Solution()
    a = S.convertToDemical(4, 4, [1, 2, 3, 3])
    print(a)

UP = 0
LEFT = 1
UP_LEFT = 2
# 记录最大共同子串长度，c[i][j]记录长度为i的x序列和长度为j的y序列的最大共同子串的长度
C = []
# 用于记录相关信息，它会用于构造最大的共同子串
B = []


def getLongestCommonStringLength(X, Y):
    """
    获取最大公共子串
    :param X:
    :param Y:
    :return:
    """
    m = len(X)
    n = len(Y)
    # 这里加1是因为算法描述中字符的下标是从1开始的，
    for i in range(0, m + 1):
        C[i].append([])
        B[i].append([])
        for j in range(0, n + 1):
            C[i].append(0)

