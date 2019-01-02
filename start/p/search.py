"""查找算法"""
from collections import deque


class Search:
    def binary_search(self, list, item):
        queue = deque()
        queue += 'aaaa'
        low = 0
        height = len(list) - 1
        while low <= height:
            mid = int(height / 2)
            guess = list[mid]
            if item == guess:
                return mid
            elif guess > item:
                height = mid - 1
            elif guess < item:
                low = mid + 1
        return None
