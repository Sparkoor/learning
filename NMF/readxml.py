"""
解析dblp.xml
"""
import xml.sax
import os
from commonUtils.Loggings import Logger

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
        # if len(_articles)==100:

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
    with open(filename, 'a') as f:
        for st in ls:
            f.write(st + '\n')


def main():
    # 创建一个xmlRead
    parser = xml.sax.make_parser()
    # turn off namepsaces
    parser.setFeature(xml.sax.handler.feature_namespaces, 0)

    Handler = ReadXML()
    parser.setContentHandler(Handler)
    parser.parse(r'D:\work\learning\NMF\datasets\dblplitter.xml')
    articles = len(_articles)
    print(articles)
    for i in _articles:
        print(i.title)
    num = 0
    # for m in _articles:
    #     ls = list_to_edges(m.author, m.date)
    #     write_list_to_file(r'D:\work\learning\NMF\datasets\dblplitters.txt', ls)
    #     num += 1
    #     if num % 1000 == 0:
    #         logger.warning("保存数据的比例{}".format(num / articles))


if __name__ == '__main__':
    main()
