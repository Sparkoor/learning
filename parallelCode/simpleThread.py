from threading import Thread
from time import sleep


class CookBook(Thread):
    def __init__(self):
        Thread.__init__(self)
        self.message = 'this is python cookbook'

    def print_message(self):
        print(self.message)

    def run(self):
        print('thread starting')
0