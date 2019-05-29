"""
psutils测试
"""
import os
import psutil
from multiprocessing import cpu_count
from concurrent.futures import ProcessPoolExecutor
from commonUtils.Loggings import Logger

logger = Logger.getLogger()

p = psutil.Process(os.getpid())

print("进程名称", p.pid)
print("使用进程", os.getpid())
logger.warning("使用进程{}".format(os.getpid()))
logger.warning("cpu:{}".format(cpu_count()))


def a(s):
    print("ssss")


def b():
    with ProcessPoolExecutor(max_workers=cpu_count()) as executor:
        executor.map(a, [1, 2, 3, 4, 5])


if __name__ == '__main__':
    # b()
    print("aa" + str(os.getpid()))
    print("{}.{}.{}".format('aa', 'bb', str(os.getpid())))
