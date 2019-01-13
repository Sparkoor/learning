aa = (1, 2, 3, 4)
print(type(aa))
bb = {1, 2, 3, 4, 5}
print(type(bb))
cc = set([1, 2])
print(cc)
# dict是这样定义的
dd = dict(na='a', d='c')
print(dd['d'])

tree = {'root': {'chil1': {'grandson1': 'maxiu', 'grandson2': 'liyliy'}, 'child2': 'jerry'}}
print(tree.keys())
print(tree.values())
tree2 = list(tree.values())
print(tree2)
print(tree2[0].keys())
tree['root2'] = {}
print(tree)
firstChildName = list(tree.keys())[0]
print(type(firstChildName))
firstChild = tree.get(firstChildName)
firstChild['grandson3'] = 'big'
print(tree)
# 相当于匿名函数,不能体现出赋值
add = lambda x: 10
print(add(1))

import logging as log


def char2num(s):
    log.info("aaa", s)
    print('进入方法' + s)
    digits = {'0': 0, '1': 1, '2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7, '8': 8, '9': 9}
    return s + str(digits[s])


if __name__ == "__main__":
    # 通过map传参
    aa = map(char2num, '023')
    print(list(aa))
