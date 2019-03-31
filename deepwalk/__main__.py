"""
deepwalk主函数
"""
import os
import sys
import random
from io import open
# todo:这个函数有点像java中的Property
from argparse import ArgumentParser, FileType, ArgumentDefaultsHelpFormatter

from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from commonUtils.Loggings import Logger

from . import graph
from . import walks as serialized_walk
from gensim.models import Word2Vec
from .Skipgram import Skipgram

# six库

import psutil
from multiprocessing import cpu_count

# todu:获取进程号？？？？
p = psutil.Process(os.getpid())

try:
    # 设置cpu相关联？？？？
    p.set_cpu_affinity(list(range(cpu_count())))
except AttributeError:
    try:
        p.cpu_affinity(list(range(cpu_count())))
    except AttributeError:
        pass

logger = Logger().getLogger()


