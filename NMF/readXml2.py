"""
使用lxml
"""
from lxml import etree
from readxml import Article
from commonUtils.Loggings import Logger

logger = Logger.getLogger()


# 会把整个文件加载成树，大文件效率低下
# doc = etree.parse(r"test.xml")

# 使用目标解析器方法
class TitleTarget(object):
    def __init__(self):
        self.text = []
        self.currentTag = ""
        self.article = None
        self.num = 0

    def start(self, tag, attrib):
        """
        在元素打开时触发。数据和元素的子元素仍不可用。
        :param tag:
        :param attrib:
        :return:
        """
        self.currentTag = tag
        if self.currentTag == 'article':
            self.article = Article()
            self.article.date = attrib['mdate']
            self.text.append(self.article)

    def end(self, tag):
        """
        在元素关闭时触发。所有元素的子节点，包括文本节点，现在都是可用的
        :param tag:
        :return:
        """
        self.currentTag = ""
        self.num += 1
        if self.num % 1000 == 0:
            logger.info("读取数据{}个".format(self.num))

    def data(self, data):
        """
        触发文本子节点并访问该文本。
        :param data:
        :return:
        """
        if self.currentTag == 'author':
            self.article.author.append(data)
        elif self.currentTag == 'title':
            self.article.title = data
        elif self.currentTag == 'journal':
            self.article.journal = data

    def close(self):
        """
        在解析完成后触发。
        :return:
        """
        return self.text


def itertest():
    parser = etree.XMLParser(target=TitleTarget())

    # This and most other samples read in the Google copyright data
    infile = r'D:\work\learning\NMF\datasets\dblplitter.xml'

    results = etree.parse(infile, parser)

    # When iterated over, 'results' will contain the output from
    # target parser's close() method
    print(len(results))
    for i, r in enumerate(results):
        print(r.title)
    # out = open(r'D:\work\learning\NMF\datasets\titles.txt', 'w')
    # out.write('\n'.join(str(results)))
    # out.close()


if __name__ == '__main__':
    itertest()
