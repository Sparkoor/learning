import feedparser

"""
feedparser的使用
"""
d = feedparser.parse('http://feed.cnblogs.com/blog/sitehome/rss')
print(d['entries'])
strList = ['AA', 'aa', 'bbb']
newStrList = [str.lower() for str in strList if len(str) > 0]
strDict = {'1': 2, '6': 3, '4': 1}
# 字典排序 返回的是key的list 为什么1会过界
sortedD = sorted(strDict, key=lambda pair: pair[0], reverse=True)
print(sortedD)
import operator

# 返回的是字典的list
sortedD2 = sorted(strDict.items(), key=operator.itemgetter(1), reverse=True)
print(sortedD2)
