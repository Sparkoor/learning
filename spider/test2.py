import requests
import json
import re
import os


class BaiduImage(object):

    def __init__(self):
        super(BaiduImage, self).__init__()

        self.page = 60  # 当前页数
        if not os.path.exists(r'./image'):
            os.mkdir(r'./image')

    def request(self):
            while True:
                request_url = 'https://image.baidu.com/search/index?tn=baiduimage&ct=201326592&lm=-1&cl=2&ie=gb18030&' \
                              'word=%C8%E9%CF%D9%B0%A9%EE%E2%B0%D0%CD%BC%CF%F1&fr=ala&ala=1&alatpl=adress&pos=0&hs=2&xthttps=111111'
                headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 6.1; WOW64; rv:40.0) Gecko/20100101 Firefox/40.0',
                           'Content-type': 'test/html'}

                response = requests.get(request_url, headers=headers)

                if response.status_code == 200:
                    data = response.text
                    print(data)


    def download(self, data):

        for d in data:

            url = d['objURL']

            pattern = re.compile(r'.*/(.*?)\.jpg', re.S)
            print('pattern', pattern)
            item = re.findall(pattern, url)

            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/56.0.2924.87 Safari/537.36"}
            response = requests.get(url, headers=headers, stream=True)
            print(response.text)
            FileName = str('image/') + item[0] + str('.jpg')

            with open(FileName, "wb") as op:
                for chunk in response.iter_content(128):
                    op.write(chunk)


if __name__ == '__main__':
    bi = BaiduImage()
    bi.request()