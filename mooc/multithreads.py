from queue import Queue
import random
import threading
import time


class Producer(threading.Thread):
    """
    生产线程
    """

    def __init__(self, t_name, queue):
        """

        """
        threading.Thread.__init__(self, name=t_name)
        self.data = queue

    """
    run方法和start方法
    """

    def run(self) -> None:
        for i in range(5):
            print("将线程{}装入".format(self.getName()))
            self.data.put(i)
            time.sleep(5)
            print('装入完成')


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
            print('消费者已经消费')
            time.sleep(5)


def main():
    """
    主线程
    :return:
    """
    queue = Queue()
    producer = Producer('pro.', queue)
    consumer = Consumer('Con.', queue)
    producer.start()
    consumer.start()
    producer.join()
    consumer.join()


if __name__ == '__main__':
    main()
