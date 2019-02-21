# import operator
#
# classCount = {'a': 1, 'b': 3}
# print(type(classCount))
#
# aa = sorted(classCount.items(), key=operator.itemgetter(0), reverse=True)
# print(aa)
# print(aa[0][0])
# print(aa[1][0])
# print(aa[0][1])
def fun1():
    fun1.a=1
    print(fun1.a)


def fun2():
    fun1.a = 2
    print(fun1.a)


if __name__ == '__main__':
    fun2()
    fun1()