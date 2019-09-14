"""
解析dblp.xml
"""
import xml.sax
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


def main():
    # 创建一个xmlRead
    parser = xml.sax.make_parser()
    # turn off namepsaces
    parser.setFeature(xml.sax.handler.feature_namespaces, 0)

    Handler = ReadXML()
    parser.setContentHandler(Handler)
    parser.parse(r'D:\work\learning\NMF\datasets\dblplitter.xml')
    for m in _articles:
        print(m.title)


if __name__ == '__main__':
    main()
