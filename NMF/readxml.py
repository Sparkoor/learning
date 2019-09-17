"""
解析dblp.xml
"""
import xml.sax
import os
from commonUtils.Loggings import Logger
import time
import math
import pandas as pd
import datetime as dt

logger = Logger.getLogger()

_articles = []
_num = 0


class Article(object):
    def __init__(self):
        self.author = []
        self.title = ""
        self.journal = ""
        self.ee = ""
        self.year = ""
        self.date = ""


class ReadXML(xml.sax.ContentHandler):
    def __init__(self):
        self.currentData = ""
        self.article = None
        self.num = 0

    def startElement(self, tag, attribute):
        """
        元素开始
        :param tag:
        :param attribute:
        :return:
        """
        self.currentData = tag
        if tag == 'article':
            self.article = Article()
            self.article.date = attribute['mdate']

    def endElement(self, tag):
        """

        :param tag:
        :return:
        """
        # 这里面没做任何处理
        # logger.info("-----------end a element--------------")
        self.currentData = ''
        if tag == 'article':
            self.num += 1
            if self.num % 100 == 0:
                logger.info("读取第{}条".format(self.num))
            _articles.append(self.article)

    def characters(self, content):
        """

        :param content:
        :return:
        """
        if self.currentData == 'author':
            self.article.author.append(content)
        elif self.currentData == 'title':
            self.article.title = content
        elif self.currentData == 'journal':
            self.article.journal = content


def list_to_edges(ls, date):
    """
    把list中的数据两两相连,还可以在这里添加规则
    :param ls:
    :param date: 产生边的时间
    :return:
    """
    lss = []
    ls_len = len(ls)
    if ls_len == 1:
        return ["{},{},{},{}".format(ls[0], None, 1, date)]
    # 太多的处理不过来
    if ls_len > 20:
        return None
    for i in range(ls_len):
        author1 = ls[i]
        for l in range(i + 1, ls_len):
            s = "{},{},{},{}".format(author1, ls[l], 1, date)
            lss.append(s)
    return lss


def write_list_to_file(filename, ls):
    """

    :param filename:
    :return:
    """
    logger.info("执行写入")
    with open(filename, 'a') as f:
        for st in ls:
            l = list_to_edges(st.author, st.date)
            for i in l:
                f.write(i + '\n')


def write_list_to_file2(filename, ls):
    """

    :param filename:
    :return:
    """
    logger.info("执行写入")
    with open(filename, 'a') as f:
        for st in ls:
            f.write(st + '\n')


# def time_slice(filename, ls, date):
#     """
#     按时间切片
#     :return:
#     """
#     with open(filename, 'a') as f:
#         for st in ls:
#             date2 = dt.datetime.strptime(st.date, "%Y-%m-%d")
#             if date2 < date:
#                 l = list_to_edges(st.author, st.date)
#                 for i in l:
#                     f.write(i + '\n')


def main():
    start = time.time()
    # 创建一个xmlRead
    parser = xml.sax.make_parser()
    # turn off namepsaces
    parser.setFeature(xml.sax.handler.feature_namespaces, 0)

    Handler = ReadXML()
    parser.setContentHandler(Handler)
    parser.parse(r'D:\workspace\pproject\NMF\analysisData\dblp.xml')
    end = time.time()
    print("use time --{}".format(end - start))
    articles = len(_articles)

    pagesize = 20000
    # 分页公式
    totalPage = (articles + pagesize - 1) / pagesize
    num = 0
    totalPage = math.ceil(totalPage)
    # todo:是用户名太多
    # todo：出现卡死的情况，应该是io打开关闭的次数太多。。。。
    # for page in range(totalPage):
    #     start = page * pagesize
    #     end = (page + 1) * pagesize
    #     if end > articles:
    #         end = articles
    #     articleBatch = _articles[int(start):int(end)]
    #     with open(r'D:\workspace\pproject\NMF\analysisData\data' + str(page), 'w') as f:
    #         for i in articleBatch:
    #             f.write(str(i.author) + "," + i.date + "\n")
    #     write_list_to_file(r'D:\workspace\pproject\NMF\analysisData\dblplitters.txt', articleBatch)
    for l in range(0, 4):
        timestarp = "201{}-12-30".format(l)
        date = dt.datetime.strptime(timestarp, "%Y-%m-%d")
        for m in _articles:
            # todo:分边的时候出现了问题
            d = dt.datetime.strptime(m.date, "%Y-%m-%d")
            if d > date:
                continue
            ls = list_to_edges(m.author, m.date)
            if ls is None:
                continue
            write_list_to_file2(r'D:\workspace\pproject\NMF\analysisData\dblplitter' + timestarp + '.txt', ls)
            num += 1
            if num % 10000 == 0:
                logger.warning("保存数据的比例{}".format(num / articles))


if __name__ == '__main__':
    main()
