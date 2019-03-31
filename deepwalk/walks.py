from commonUtils.Loggings import Logger

from io import open
from os import path
from time import time
from multiprocessing import cpu_count
import random
from concurrent.futures import ProcessPoolExecutor
# todo：Counter的用法
from collections import Counter
# six的zip

from . import graph

logger = Logger().getLogger()
__current_graph = None
# 加速字符串编码
__vertex2str = None


def count_words(file):
    """
    计算单词在句子中出现的频率
    这是并行帮助函数
    :param file:
    :return:
    """
    c = Counter()
    logger.info("c init num {}".format(c))
    with open(file, 'r') as f:
        for l in f:
            words = l.strip().split()
            c.update(words)
    logger.info("c update num {}".format(c))
    return c


def count_textfiles(files, workers=1):
    """
    计算单词在整个文档集中的概率
    :param files:
    :param workers:
    :return:
    """
    c = Counter()
    with ProcessPoolExecutor(max_workers=workers) as executor:
        for c_ in executor.map(count_words, files):
            c.update(c_)
    return c


def count_lines(f):
    """
    计算文章的行数
    :param f:
    :return:
    """
    if path.isfile(f):
        num_line = sum(1 for line in open(f))
        return num_line
    else:
        return 0


def _write_walks_to_disk(args):
    """
    保存每次游走生成的句子
    :param args:
    :return:
    """
    num_paths, path_length, alpha, rand, f = args
    G = __current_graph
    with open(f, 'w') as fout:
        for walk in graph.buildDeepWalkCorpusIter(G=G, numLength=num_paths, pathLength=path_length, alpha=alpha,
                                                  rand=rand):
            fout.write(u"{}\n".format(u" ".join(v for v in walk)))
    logger.info('把随机游走的句子保存到文件')

    return f


def write_walks_to_disk(G, filebase, num_paths, path_length, alpha=0, rand=random.Random(0), num_workers=cpu_count(),
                        always_rebuild=True):
    """

    :param G:
    :param filebase:
    :param num_paths:
    :param path_length:
    :param alpha:
    :param rand:
    :param num_workers:
    :param always_rebuild:
    :return:
    """
    global __current_graph
    __current_graph = G
    # todo:不明确用途
    files_list = ["{}.{}".format(filebase, str(x)) for x in list(range(num_paths))]
    excepted_size = len(G)
    args_list = []
    files = []

    if num_paths <= num_workers:
        # 会循环出个全是1的列表
        paths_per_work = [1 for x in range(num_paths)]
    else:
        # 输出每个cpu分配的任务数，todo：分段读取，grouper类似分页
        paths_per_work = [len(list(filter(lambda z: z != None, [y for y in x]))) for x in
                          graph.grouper(int(num_paths / num_workers) + 1, range(1, num_paths + 1))]
        logger.error("文件数量大于核数是，任务的分配{}".format(paths_per_work))

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        for size, file_, ppw in zip(executor.map(count_lines, files_list), files_list, paths_per_work):
            # todo:这两个变量不是很理解
            if always_rebuild or size != (ppw * excepted_size):
                args_list.append((ppw, path_length, alpha, random.Random(rand.randint(0, 2 ** 31)), file_))
            else:
                files.append(file_)
    # 应该是保存之后，返回文件名称路径吧
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        for file_ in executor.map(_write_walks_to_disk, args_list):
            files.append(file_)
    logger.error("返回的应该是文件路径吧{}".format(files))
    return files


class WalksCorpus(object):
    def __init__(self, file_list):
        self.file_list = file_list

    # 迭代方法
    def __iter__(self):
        for file in self.file_list:
            with open(file, 'r') as f:
                for line in f:
                    # todo：需要调用next()才能继续执行吧
                    yield line.split()


def combine_files_iter(file_list):
    """
    todo:为什么使用yield还需要理解
    :param file_list:
    :return:
    """
    for file in file_list:
        with open(file, 'r') as f:
            for line in f:
                yield line.split()
