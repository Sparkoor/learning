"""
使用concurrent.futures 测试一下多线程
"""
import os
from NMF.iterTest import loadFiles
# 进程池 cpu密集型
from concurrent.futures import ProcessPoolExecutor
# 线程池 I/O密集型
from concurrent.futures import ThreadPoolExecutor

total = []


def read_file(filename):
    ls = []
    with open(filename) as f:
        for i, m in enumerate(f):
            ls.append(m)
        print("来自文件{}------长度{}".format(filename, len(ls)))
    total.append(ls)


def threadt(absoultepath):
    with ThreadPoolExecutor() as executor:
        future = executor.map(read_file, loadFiles(absoultepath))
        print("执行结束")


def processt(ls):
    with ThreadPoolExecutor() as executor:
        futures=executor.submit()


if __name__ == '__main__':
    abs = os.path.abspath(r'D:\workspace\pproject\NMF\analysisData\data')
    threadt(abs)
    print(len(total))
