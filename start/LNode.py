class LNode:
    ##next_ 是为了避免和python的标准函数重名
    def __init__(self,elem,next_=None):
        self.elem=elem
        self.next=next_
        

