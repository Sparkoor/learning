"""
多线程的练习
"""
import threading
from termcolor import *
from commonUtils.Loggings import *

logger = Logger().getLogger()


def printMessage(i):
    print(colored("多线程{}".format(i), color='red'))
    return


def function1():
    threads = []
    for i in range(5):
        t = threading.Thread(target=printMessage, args=(i,))
        threads.append(t)
        # 执行该方法线程才会开始执行
        t.start()
        # join()会导致调用线程等待，直到它执行完毕
        t.join()


def threadingWith(statement):
    # 上下文管理器
    with statement:
        logger.info("%s acquired directly" % statement)


def threadingNotWith(statement):
    statement.acquire()
    try:
        logger.debug("%s acquire directly" % statement)
    finally:
        statement.release()


def testWithCombine():
    """
    测试with
    :return:
    """
    # 使用lock
    lock = threading.Lock()
    # 使用RLock
    rLock = threading.RLock()
    # 使用条件
    condition = threading.Condition()
    # 使用信号量
    mutex = threading.Semaphore()
    threadingSynchronizationList = [lock, rLock, condition, mutex]
    # 循环调用各种方式
    for statement in threadingSynchronizationList:
        t1 = threading.Thread(target=threadingWith, args=(statement,))
        t2 = threading.Thread(target=threadingNotWith, args=(statement,))
        t1.start()
        t2.start()
        t1.join()
        t2.join()


if __name__ == '__main__':
    testWithCombine() 
