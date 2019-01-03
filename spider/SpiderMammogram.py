"""
爬取网上乳腺癌钼靶图像
"""

import requests
import time
import re


def spider(keyword):
    for page in range(1, 18):
        # 这个url只能用来下载相关乳腺癌钼靶图像，需要使劲的分析请求
        url = 'https://image.baidu.com/search/acjson?tn=resultjson_com&ipn=rj&ct=201326592&' \
              'is=&fp=result&queryWord=%E4%B9%B3%E8%85%BA%E7%99%8C%E9%92%BC%E9%9D%B6%E5%9B%BE%E5%83%8F&cl=2&' \
              'lm=-1&ie=utf-8&oe=utf-8&adpicid=&st=&z=&ic=&hd=&latest=&copyright=&' \
              'word=' + keyword + \
              'height=&face=&istype=&qc=&nc=&fr=&expermode=&force=&pn=' + str(page * 30) + '&rn=30&gsm=1e&154651709902'
        response = download(url)
        analyHtml(response.text)
        time.sleep(3)


header = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)\
            AppleWebKit/537.36 (KHTML, like Gecko) \
            Chrome/71.0.3578.98 Safari/537.36'
}


def download(url):
    return requests.get(url=url, headers=header)


def analyHtml(html):
    # 根据返回的数据进行分析，用提取图片链接
    p = re.compile("thumbURL.*?\.jpg")
    # 获取正则匹配结果，返回的是一个list
    s = p.findall(html)
    index = 1
    for i in s:
        url = i.replace("thumbURL\":\"", "")
        image_saver(url, 'image' + str(index))
        print('下载图片：' + i)
        index = index + 1


# 保存图片
def image_saver(url, name):
    img = requests.get(url=url, headers=header)
    with open(name + '.jpg'.format(), 'wb') as f:
        f.write(img.content)


if __name__ == '__main__':
    spider('乳腺癌钼靶图像')
