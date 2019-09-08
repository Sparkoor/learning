"""
把空手到网络构造成带时间戳的
"""
from collections import defaultdict


def load_karate(filename1,filename2):
    graph = defaultdict()
    with open(filename1) as f:
        for line in f:
            s = line.strip().split(" ")
            key = s[0]
            for v in s[1:]:
                graph[key].append(v)

    # with open(filename2,'r') as f:
    #     for