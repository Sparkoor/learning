# %%

import pandas as pd
import numpy as np
import jieba
import re

articles = pd.read_csv(r'D:\workspace\pproject\nplearn\sample\Train_DataSet.csv', encoding='utf-8').astype(str)
print(articles.shape)
print(type(articles.values))
# 先分词，统计词的总数，统计词频，对词
# 构造文本库
datasets = pd.DataFrame(columns=('id', 'title', 'content'))
for i in articles.values:
    title = jieba.lcut(i[1])
    content = jieba.lcut(i[2])
    datasets = datasets.append([{'id': i[0], 'title': title, 'content': content}])
# print(datasets)
datasets.to_csv(r'D:\workspace\pproject\nplearn\sample\Train_DataSets.csv', index=True)
# jieba库的使用
cut = jieba.lcut('中华人民共和国国歌')


# print(cut)

def write_to_file(filename):
    """
    把分词结果写入文件
    :param filename:
    :return:
    """
    pass


def clear_data(str):
    """
    数据清理：清理分词得到的空格英文字母等
    :param str:
    :return:
    """
    pattern = re.compile('[,.?，。？]')
    ss = pattern.match("woshi ,uoe.xiix.djdjd")