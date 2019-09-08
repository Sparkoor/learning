import pandas as pd
import numpy as np
import jieba
import re
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from commonUtils.Loggings import Logger
# todo:获取路径要用
import os

logger = Logger.getLogger()


class Emotion(object):
    def __init__(self, args):
        self.args = args
        self.articles = pd.read_csv(self.args.data_path1, encoding='utf-8').astype(str)

    # print(type(articles.values))
    # # 先分词，统计词的总数，统计词频，对词
    # # 第一是需要取得值，后面几项是过滤条件
    # aa = articles['title'][articles['id'] == '7a3dd79f90ee419da87190cff60f7a86']

    @staticmethod
    def clear_noncn(str):
        # 过滤掉标点符号
        reg_str = '[a-zA-z0-9,!"#$%&\'()*+，./;：；|<=>?、\[\]^？:。！”●•▽“\-（）]+'
        # 过滤掉非汉字词只用这一个就行了
        reg_str2 = "[^\u4E00-\u9FA5]"
        str = re.sub(reg_str2, '', str)
        return str

    # def count_word(set,)
    @staticmethod
    def write_to_file(filename, set, num=0):
        """
        把分词结果写入文件
        :return:
        """
        logger.info("写入文件....")
        with open(filename, 'w', encoding='utf-8') as f:
            f.write("{}\n".format(num))
            for word in set:
                f.write(word + ' ')

    def make_words(self):
        # articles = pd.read_csv(self.args.data_path1, encoding='utf-8').astype(str)
        print(self.articles.shape)
        # 构造文本库
        datasets = pd.DataFrame(columns=('id', 'title', 'content'))
        art_length = len(self.articles.values)
        trans_num = 0
        for i in self.articles.values:
            title = self.clear_noncn(i[1])
            # title = re.sub(reg_str2, '', title)
            title = jieba.lcut(title)
            content = self.clear_noncn(i[2])
            # content = re.sub(reg_str2, '', content)
            content = jieba.lcut(content)
            datasets = datasets.append([{'id': i[0], 'title': title, 'content': content}])
            trans_num += 1
            if trans_num % 200 == 0:
                logger.info("转换数据{}".format(trans_num / art_length))
                break
        # print(datasets)
        datasets.to_csv(self.args.data_save1, index=False)

    def make_corpus(self):
        """
        构造语料库
        :return:
        """
        # jieba库的使用
        # cut = jieba.lcut('中华人民共和国国歌')
        # 统计词库
        article_words = pd.read_csv(self.args.data_save1, encoding='utf-8')
        title_corpus = set()
        content_corpus = set()
        articles = len(article_words)
        count_nums = 0
        for words in article_words.values:
            reg = "[\'\"\[\]]+"
            word_str = re.sub(reg, '', words[1])
            # print(word_str)
            title_words = word_str.strip().split(",")
            for word in title_words:
                title_corpus.add(word)
            word_str = re.sub(reg, '', words[2])
            content_words = word_str.strip().split(",")
            for word in content_words:
                content_corpus.add(word)
            count_nums += 1
            if count_nums % 1000 == 0:
                logger.info("统计词的进度,完成了{}".format(count_nums / articles))
        title_corpus_len = len(title_corpus)
        logger.info("统计出title词有{}个".format(title_corpus_len))
        content_corpus_len = len(content_corpus)
        logger.info("统计出content词有{}个".format(content_corpus_len))
        self.write_to_file(self.args.title_corpus_file, title_corpus, title_corpus_len)
        logger.info("title语料库构造完成")
        self.write_to_file(self.args.content_corpus_file, content_corpus, content_corpus_len)
        logger.info("content语料库构造完成")
        # print(cut)


def main():
    parse = ArgumentParser("emotion", formatter_class=ArgumentDefaultsHelpFormatter, conflict_handler="resolve")
    # note:dest这个可有可无,有了调用时一定要用dest里的名称
    parse.add_argument("--data-path1", default=r'D:\work\learning\nplearn\datasets\Train_DataSet.csv',
                       type=str, help="data path")
    parse.add_argument("--data-save1", default=r'D:\work\learning\nplearn\datasets\Train_DataSets.csv', type=str,
                       help="保存分词完成的文件")
    parse.add_argument("--title-corpus-file", default=r'D:\work\learning\nplearn\datasets\titlecorpus.txt')
    parse.add_argument("--content-corpus-file", default=r'D:\work\learning\nplearn\datasets\contentcorpus.txt')
    args = parse.parse_args()
    E = Emotion(args)
    # 先分词
    # E.make_words()
    # 统计语料库
    E.make_corpus()
    # 将每条文章弄成词频向量


if __name__ == '__main__':
    main()
