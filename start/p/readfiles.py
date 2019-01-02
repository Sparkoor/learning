# 从标准库导入os
import os
import sys
import pickle

# 判断当前工作目录
work_path = os.getcwd()
print('工作目录' + work_path)

data = open('data.txt')
if 'data' in locals():
    data.close()
print(data.readline(), end='')
print(data.readline(), end='')
# 从文件最头
data.seek(0)
print(data.readline(), end='')
data.seek(3)
print(data.readline(), end='')

data.seek(0)
for d in data:
    print(d, end='')
    if not d.split(':') == -1:
        (st1, str2) = d.split(':')
        print(st1 + 'said')
        print(str2)

data.close()
# 一场处理
try:
    with open('data.txt', 'w') as data:
        print("aaa", file=data)
except IOError as err:
    print(str(err))


# 使用pickle保存数据
def solution1(self, man, fn=sys.stdout):
    try:
        with open('data.txt', 'wb') as data:
            pickle.dump('aaaa', data)
    except IOError as err:
        print(str(err))
    except pickle.PickleError as perr:
        print(str(perr))
