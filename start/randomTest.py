"""
用于测试random
"""
import random

l = [2, 3, 4, 5, 6, 4]

random.Random().shuffle(l)
print(l)
rand = random.Random()
s=rand.choice(l)
print(s)
