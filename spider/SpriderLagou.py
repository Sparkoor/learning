"""
爬取拉钩网的招聘信息测试
"""
import requests
from lxml import etree
import csv
import time
from multiprocessing import pool
# 日志打印以后再学
import logging


# 写入文件
def csv_write(item):
    with open('lagou.csv', encoding='gbk', newline='') as csvfile:
        writer = csv.writer(csvfile)
        try:
            writer.writerow(item)
        except Exception as e:
            print('保存文件时出错!!!', e)
            logging.exception(e)


# 获取网页内容
def spider(url):
    header = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)\
                AppleWebKit/537.36 (KHTML, like Gecko) \
                Chrome/71.0.3578.98 Safari/537.36'
    }
    res = requests.get(url=url, headers=header)
    return res.text


# 处理获取的文本信息
def analyseResult(resultText):
    # 使用json解码
    jsons = resultText.json()
