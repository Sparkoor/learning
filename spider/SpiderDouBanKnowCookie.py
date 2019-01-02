# 利用cookie对需要登陆操作的网页进行爬取
# requests保持登陆

import requests
from lxml import etree


# 格式化cookie，元组形式
def coo_regular(cookie):
    cook = {}
    for v_k in cookie.split(';'):
        k, v = v_k.split('=', 1)
        cook[k.strip()] = v.replace('"', '')
    return cook


header = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)\
           AppleWebKit/537.36 (KHTML, like Gecko) \
           Chrome/71.0.3578.98 Safari/537.36'
}

cookie_str = 'bid=TQ4eHMhbqfA; gr_user_id=cf8c99e6-317f-4601-89a6-8583cbd6c165;\
vwo_uuid_v2=D1D5548A239BC92BD4EDC6E7AF3A57A42|ac823a60ca629062382a75b3392aa604;' \
             'douban-fav-remind=1; ll="118123"; viewed="1131829_2075878_3258245"; ps=y; _' \
             'pk_ref.100001.8cb4=%5B%22%22%2C%22%22%2C1546435158%2C%22https%3A%2F%2Fwww.google.com.hk%2F%22%5D; ' \
             '_pk_ses.100001.8cb4=*; __utma=30149280.1940035125.1535851191.1546433071.1546435159.10; __utmc=30149280; _' \
             '_utmz=30149280.1546435159.10.10.utmcsr=google|utmccn=(organic)|utmcmd=organic|utmctr=(not%20provided); ' \
             '_ga=GA1.2.1940035125.1535851191; _gid=GA1.2.1875739074.1546435788; ue="884831651@qq.com"; dbcl2="189532957:4ZxNMyc7Obs";' \
             ' ck=_bEv; ap_v=0,6.0; push_noty_num=0; push_doumail_num=0; __utmt=1; __utmv=30149280.18953; douban-profile-remind=1; ' \
             '_pk_id.100001.8cb4=592dbaa45d686e40.1535964885.6.1546435939.1546433068.; __utmb=30149280.8.10.1546435159'
cookie = coo_regular(cookie_str)


# 登陆地址：https://www.douban.com/accounts/login

def spider(url):
    response = requests.post(url=url, headers=header, cookies=cookie)
    print('zwp2019的帐号' in response.text)


if __name__ == '__main__':
    url = 'https://www.douban.com/'
    spider(url)
