from queue import Queue
import random
import threading
import time
from NMF.iterTest import loadFiles


# 这个多线程不太对
class Producer(threading.Thread):
    """
    生产线程
    """

    def __init__(self, t_name, queue, filepath):
        """

        """
        threading.Thread.__init__(self, name=t_name)
        self.data = queue
        self.path = filepath

    """
    run方法和start方法
    """

    def run(self) -> None:
        for i in loadFiles(self.path):
            print("将线程{}.{}装入".format(i, self.getName()))
            self.data.put(i)
            print('装入完成')
            time.sleep(1)
            print("-------------------队列中有{}".format(self.data.maxsize))


class Consumer(threading.Thread):
    """
    线程消费者
    """

    def __init__(self, t_name, queue):
        threading.Thread.__init__(self, name=t_name)
        self.data = queue

    def run(self) -> None:
        for i in range(5):
            val = self.data.get()
            print('消费者已经消费{},还有{}'.format(val, i))
            time.sleep(1)


def main():
    """
    主线程
    :return:
    """
    queue = Queue()
    producer = Producer('pro.', queue, r'D:\workspace\pproject\NMF\analysisData\data')
    consumer = Consumer('Con.', queue)
    producer.start()
    consumer.start()
    producer.join()
    consumer.join()


if __name__ == '__main__':
    main()
    # for j in loadFiles(r'D:\workspace\pproject\NMF\analysisData\data'):
    #     print(j)
