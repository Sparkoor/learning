import pandas as pd
import numpy as np
import jieba
import re
from commonUtils.Loggings import Logger

logger = Logger.getLogger()

articles = pd.read_csv(r'D:\work\learning\nplearn\datasets\Train_DataSet.csv', encoding='utf-8').astype(str)
print(articles.shape)
print(type(articles.values))
# 先分词，统计词的总数，统计词频，对词
# 第一是需要取得值，后面几项是过滤条件
aa = articles['title'][articles['id'] == '7a3dd79f90ee419da87190cff60f7a86']
print(aa.values)
print(aa.shape)


def clear_noncn(str):
    # 过滤掉标点符号
    reg_str = '[a-zA-z0-9,!"#$%&\'()*+，./;：；|<=>?、\[\]^？:。！”●•▽“\-（）]+'
    # 过滤掉非汉字词只用这一个就行了
    reg_str2 = "[^\u4E00-\u9FA5]"
    str = re.sub(reg_str2, '', str)
    return str


# 构造文本库
datasets = pd.DataFrame(columns=('id', 'title', 'content'))
art_length = len(articles.values)
trans_num = 0
for i in articles.values:
    title = clear_noncn(i[1])
    # title = re.sub(reg_str2, '', title)
    title = jieba.lcut(title)
    content = clear_noncn(i[2])
    # content = re.sub(reg_str2, '', content)
    content = jieba.lcut(content)
    datasets = datasets.append([{'id': i[0], 'title': title, 'content': content}])
    trans_num += 1
    if trans_num % 200 == 0:
        logger.info("转换数据{}".format(trans_num / art_length))
        break
# print(datasets)
datasets.to_csv(r'D:\work\learning\nplearn\datasets\Train_DataSets.csv', index=False)
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

# if __name__ == '__main__':
#     pass
