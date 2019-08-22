# 引入线程池
from multiprocessing.dummy import Pool as pl
import requests
# 存数据的
import csv
# time可以睡眠
import time



def spider(a):
    url = 'http://www.1kkk.com/manhua10684/'
    data_list = []
    for i in range(1, 2):
        if i == 1:
            response = download(url)
            analyMainHtml(response.text)
            time.sleep(3)
        else:
            response = download(url + 'pg' + i)
            analyMainHtml(response.text)
            time.sleep(3)


header = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)\
            AppleWebKit/537.36 (KHTML, like Gecko) \
            Chrome/71.0.3578.98 Safari/537.36'
}


def download(url):
    # 链家的第一页是不带页码的所以在循环中判断
    return requests.get(url=url, headers=header)


def analyHtml(html):

    # selector = etree.HTML(html)
    print(html)
    # //*[@id="content"]/div[1]/div[1]/div[1]/div/p[1]/a
    # //*[@id="detail-list-select-1"]
    # 获取图片地址信息
    imgUrlList = selector.xpath('//*[@id="detail-list-select-1"]/ul/li/a/@href')
    urlList = []
    for i in imgUrlList:
        print(i)


def analyPgHtml(url):
    resp = download(url)
    selector = etree.HTML(resp.text)


def spiderMain():
    s='a'

# 保存到文件
def data_writer(item):
    with open('lianjia.csv', 'a', encoding='utf-8', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(item)


# 保存图片
def image_saver(url, xiaoqu):
    # TODO:添加一个header
    img = requests.get(url=url, headers=header)
    # 书上有个例子没法实现
    with open('a.jpg'.format(), 'wb') as f:
        f.write(img.content)


if __name__ == '__main__':
    """
    地址没找到规律
    """
    resp = download('http://www.1kkk.com/ch142-752763/#ipg2')
    ss=etree.HTML(resp.text)
    str=ss.xpath('//*[@id="cp_image"]')
    print(str)
    print(etree.tostring(ss))
