# import re
#
# str = '[\'\n                    合租 · 恒大名都 3室1厅 600元                  \']'
# # pattern = re.compile('\[|\]')
# # ss = re.match(pattern, str)
# # print(ss.group(0))
#
# aa = str.replace('[\'', '') \
#     .replace('\']', '').strip()
# print(aa)
#
# str = 'aa=ee'
# v, k = str.split('=')
# print(v)
# print(str.split('='))

str = 'aaa222222222222'
l = str[0:6]
print(l)

import requests
from lxml import etree


def spider():
    header = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)\
                   AppleWebKit/537.36 (KHTML, like Gecko) \
                   Chrome/71.0.3578.98 Safari/537.36'
    }
    url = 'https://www.lagou.com/zhaopin/Java/1/?filterOption=1'
    res = requests.get(url=url, headers=header)
    source = etree.HTML(res.text)
    str = source.xpath('//*[@id="s_position_list"]/ul/li[1]/div[1]/div[1]/div[1]/a/h3/text()')
    print(str)


if __name__ == '__main__':
    spider()
