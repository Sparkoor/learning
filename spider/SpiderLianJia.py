"""
https://sy.lianjia.com/zufang/pg3/#contentList
爬取租房信息
"""
from lxml import etree
import requests
# 存数据的
import csv
# time可以睡眠
import time


def spider():
    header = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)\
         AppleWebKit/537.36 (KHTML, like Gecko) \
         Chrome/71.0.3578.98 Safari/537.36'
    }
    # 链家的第一页是不带页码的所以在循环中判断
    pre_url = 'https://sy.lianjia.com/zufang/'
    suf_url = '#contentList'
    for i in range(1, 2):
        if i == 1:
            response = requests.get(url=pre_url, headers=header)
            analyHtml(response.text)
            time.sleep(3)
        else:
            response=requests.get(url=pre_url+'pg'+i,headers=header)
            analyHtml(response.text)
            time.sleep(3)

def analyHtml(html):
    selector=etree.HTML(html)
    #//*[@id="content"]/div[1]/div[1]/div[1]/div/p[1]/a
    houselist=selector.xpath('//*[@id="content"]/div/div/div')
    for house in houselist:
        print(house.xpath('div/p[1]/a/text()'))

spider()

