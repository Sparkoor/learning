import xml.sax

datas = set()


class MovieHandler(xml.sax.ContentHandler):
    def __init__(self):
        self.title = ""
        self.ee = ""
        self.year = ""
        self.journal = ""

    # 元素开始事件处理
    def startElement(self, tag, attributes):
        self.CurrentData = tag
        if tag == "article":
            key = attributes["key"]

    # 元素结束事件处理
    def endElement(self, tag):
        if self.CurrentData == "title":
            print(u'title:', self.title)
        elif self.CurrentData == "ee":
            print(u'ee:', self.ee)
        elif self.CurrentData == "journal":
            print(u'journal:', self.journal)
        if self.CurrentData == "year":
            print(u'year:', self.year)
        self.CurrentData = ""

    # 内容事件处理
    def characters(self, content):
        if self.CurrentData == "title":
            self.title = content
        elif self.CurrentData == "ee":
            self.ee = content
        elif self.CurrentData == "year":
            self.year = content
        elif self.CurrentData == "journal":
            self.journal = content


if (__name__ == "__main__"):
    # 创建一个 XMLReader
    parser = xml.sax.make_parser()
    # turn off namepsaces
    parser.setFeature(xml.sax.handler.feature_namespaces, 0)

    # 重写 ContextHandler
    Handler = MovieHandler()
    parser.setContentHandler(Handler)
    parser.parse(r"D:/sample/dblp.xml")
