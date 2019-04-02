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


def debug(type_, value, tb):
    """
    todo:用途不明，但是像自定义的调试函数
    :param type_:
    :param value:
    :param tb:
    :return:
    """
    if hasattr(sys, 'ps1') or not sys.stderr.isatty():
        sys.__excepthook__(type_, value, tb)
    else:
        import traceback
        import pdb
        traceback.print_exception(type_, value, tb)
        print(u"\n")
        pdb.pm()


def process(args):
    """
    任务函数
    :param args:
    :return:
    """
    if args.format == 'adjlist':
        G = graph.loadAdjacencyList(args.input, undirected=args.undirected)
    elif args.format == 'edgelist':
        G = graph.loadEdgeList(args.input, undirected=args.undirected)
    elif args.format == 'mat':
        G = graph.loadMatfile(args.input, variableName=args.matfile_varible_name, undirected=args.undirected)
    else:
        raise Exception("unknow file format")
    logger.info("number of node:{}".format(len(G.nodes())))
    num_walks = len(G.nodes()) * args.walk_length
    logger.info("number of walks:{}".format(num_walks))
    data_size = num_walks * args.walk_length
    logger.info("number of data_size:{}".format(data_size))
    if data_size < args.max_memory_data_size:
        logger.info("walking...")
        walks = graph.buildDeepWalkCorpus(G, numPaths=args.number_walks, pathLength=args.walk_length, alpha=0,
                                          rand=random.Random(args.seed))
        logger.info("training....")
        model = Word2Vec(walks, size=args.representation_size, window=args.window_size, min_count=0, sg=1, hs=1,
                         works=args.works)
    else:
        logger.info("data size is more than memory")
        logger.info("working...")
        walks_filebase = args.output + ".walks"
        walk_files = serialized_walk.write_walks_to_disk(G, walks_filebase, num_paths=args.number_walks,
                                                         path_length=args.walk_length, alpha=0,
                                                         rand=random.Random(args.seed), num_workers=args.works)
        logger.info("counting vertex frequency...")
        if not args.vertex_freq_degree:
            vertex_counts = serialized_walk.count_textfiles(walk_files, args.workers)
        else:
            vertex_counts = G.degree(nodes=G.keys())
        logger.info("Training...")
        # 生成语料库
        walks_corpus = serialized_walk.WalksCorpus(walk_files)
        model = Skipgram(sentences=walks_corpus, vocabulary_counts=vertex_counts, size=args.representation_size,
                         window=args.window_size, min_count=0, trim_rule=None, workers=args.workers)
    model.wv.save_word2vec_format(args.output)


def main():
    """
    note:用于加载初始数据
    :return:
    """
    # 初始化
    parser = ArgumentParser("deepwalk", formatter_class=ArgumentDefaultsHelpFormatter, conflict_handler="resolve")
    parser.add_argument("--debug", dest="debug", action="store_true", default=False,
                        help="drop a debugger if exception is raised")
    parser.add_argument("--format", default="adjlist", help="file format of input file")
    # todo:不明确的用法
    parser.add_argument("--input", nargs="?", required=True, help="input graph file")
    # note:应该是调用函数
    parser.add_argument("-l", "--log", dest="log", default="INFO", help="log verbosity level")
    parser.add_argument("--matfile-variable-name", default="network",
                        help="variable name of adjacency matrix inside a .mat file.")
    parser.add_argument("--max-memory-data-size", default=1000000000, type=int,
                        help="size to start dumping walks to disk,instead of keeping them in memory")
    parser.add_argument("--number-walks", default=10, type=int, help="number of random walks to start at each node")
    parser.add_argument("--representation-size", default=64, type=int,
                        help="number of latent dimentions to learn for each node")
    parser.add_argument("--seed", default=0, type=int, help="seed for random walk generator")
    parser.add_argument("--undirected", default=True, type=bool, help="Treat graph as undirected")
    parser.add_argument("--vertex-freg-degree", default=False, action="store_true",
                        help="use vertex degree to estimate the frequency of nodes"
                             "in the random walks.this option is faster than "
                             "calculating the vocabulary")
    parser.add_argument("--walk-length", default=40, type=int, help="Length of the random walk started at each node")
    parser.add_argument("--window-size", default=5, type=int, help="window size of skipgram model.")
    parser.add_argument("--worker", default=1, type=int, help="number of parallel processes")

    args = parser.parse_args()
    numeric_level = getattr(logger, args.log.upper(), None)

    if args.debug:
        sys.excepthook = debug
    # note:调用函数传值
    process(args)


if __name__ == '__main__':
    sys.exit(main())
