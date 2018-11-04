import LNode
import LinkedListUnderflow


class LList:

    def __init__(self):
        self.head = None

    def is_Empty(self):
        return self.head is None

    def prepend(self, elem):
        self.head = elem

    def pop(self):
        if self.head is None:
            raise LinkedListUnderflow("in pop")
