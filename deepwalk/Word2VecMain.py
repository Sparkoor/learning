"""

"""
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import os

import walks as serialized_walk
from gensim.models import Word2Vec
import Skipgram as skg


# if not args.vertex_freq_degree:
#     vertex_counts = serialized_walk.count_textfiles(walk_files, args.workers)
# else:
#     vertex_counts = G.degree(nodes=G.keys())
# logger.info("Training...")
# # 生成语料库
# walks_corpus = serialized_walk.WalksCorpus(walk_files)
# model = Skipgram(sentences=walks_corpus, vocabulary_counts=vertex_counts, size=args.representation_size,
#                  window=args.window_size, min_count=0, trim_rule=None, workers=args.workers)
# model.wv.save_word2vec_format(args.output)

def main():
    parse = ArgumentParser(description="word2vec parser", formatter_class=ArgumentDefaultsHelpFormatter,
                           conflict_handler='resolve')
    parse.add_argument('--representation-size', type=int, default=100, help="向量维度")
    parse.add_argument("--window", type=int, default=10, help="窗口值")
    parse.add_argument("--min-count", type=int, default=0, help="最小的总量")
    parse.add_argument("--workers", type=int, default=4, help="电脑核数")

    args = parse.parse_args()
    process(args)


def process(args):
    basePath = "D:\workspace\pproject\deepwalk\example\\textoot1"
    files = []
    for i in range(0, 8):
        files.append(os.path.join(basePath, str(i)))

    vertex_count = serialized_walk.count_textfiles(files, 4)
    corpus = serialized_walk.combine_files_iter(files, 4)
    model = skg.Skipgram(sentences=corpus, vocabulary_counts=vertex_count, size=args.representation_size,
                         window=args.window_size, min_count=0, trim_rule=None, workers=args.workers)
    model.wv.save_word2vec_format(os.path.join(basePath, ".emb"))


if __name__ == '__main__':
    main()
