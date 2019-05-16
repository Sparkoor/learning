"""
测试defaultdict
defaultdict可以接受一个内建函数list作为参数
相当于可以为同一个key添加多个值，节省了一些步骤
"""
from collections import defaultdict

s = [('yellow', 1), ('blue', 2), ('yellow', 3), ('blue', 4), ('red', 1)]
# 接受一个对象
d = defaultdict(list)

for k, v in s:
    d[k].append(v)
# [('yellow', [1, 3]), ('blue', [2, 4]), ('red', [1])]
print(list(d.items()))

a = dict()
for k, v in s:
    # a[k].a
    # a[k] = v
    # 和这个等价
    a.setdefault(k, []).append(v)

print(a.items())
