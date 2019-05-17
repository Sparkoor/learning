"""
collections测试
"""
from collections import namedtuple

p = (1, 2, 3, 4, 5)
print(type(p))
print(p)
# 可以定义坐标之类的，并且保证不可变，Point tuple的一个子类
Point = namedtuple('name', ['x', 'y'])
print(Point)  # <class '__main__.name'> name的一个实例
print(type(Point))
p = Point(1, 2)
print(p)
q = Point(2, 3)
print(q)

# 使用list存储数据时，按索引访问元素很快，但是插入和删除元素就很慢了，
# 因为list是线性存储，数据量大的时候，插入和删除效率很低。
# deque是为了高效实现插入和删除操作的双向列表，适合用于队列和栈：
from collections import deque

l = deque(['a', 'b', 'c'])

print(type(l))
# 不是list的子类
print(isinstance(l, list))
print(type(deque))
print(isinstance(deque, list))
# 双向插入
l.append('f')
l.appendleft('g')
# 主要作用插入时，不用检查已经存在的key，当存在时直接进行插入
from collections import defaultdict

s = [('yellow', 1), ('blue', 2), ('yellow', 3), ('yellow', 3), ('blue', 4), ('red', 1)]
# 接受一个对象
d = defaultdict(list)
# 保持Key的顺序
from collections import OrderedDict
# 简单的计数器，统计字符出现的次数,key-value 的形式，是dict的子类
from collections import Counter

c = Counter()

for ch in 'programming':
    c[ch] = c[ch] + 1

print(c['r'])
