from multiprocessing import Process, Value, Lock, Manager
from multiprocessing.managers import BaseManager


class Employee(object):
    def __init__(self, name):
        self.name = name

    def getPay(self):
        return self.name


class MyManager(BaseManager):
    pass


def Manager2():
    m = MyManager()
    m.start()
    return m


MyManager.register('Employee', Employee)


def func1(em, lock):
    with lock:
        print(em.getPay())


if __name__ == '__main__':
    em = Manager2().Employee('zhangsan')
    lock = Lock()
    proces = [Process(target=func1, args=(em, lock)) for i in range(10)]
    for p in proces:
        p.start()
    for p in proces:
        p.join()
