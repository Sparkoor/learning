import requests
from lxml import etree

headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) '
                  'AppleWebKit/537.36 (KHTML, like Gecko) Chrome/71.0.3578.98 Safari/537.36'
}


def spider(url):
    # 进入登陆页
    response = requests.get(url=url, headers=headers)
    selector = etree.HTML(response.text)
    # //*[@id="lzform"]/fieldset/div[1] //*[@id="lzform"]/fieldset/input
    captch = selector.xpath('//*[@id="captcha_image"]/@src')
    if captch:  # 判断是否有验证码
        captch_url = captch[0]  # 获取验证码图片url
        captch_image = requests.get(captch_url)
        from PIL import Image  # 处理图片
        from io import BytesIO
        img = Image.open(BytesIO(captch_image.content))  # 打开验证码图片
        img.show()
        captch_text = input(u'输入验证码：')  # 输入标识的验证码
        # 解析验证码中的captcha_id
        captcha_id = captch_url.split('=')[1].split('&')[0]
    else:
        captcha_id = None
        captch_text = None

    # 构造form表单
    formData = {
        'source': 'index_nav',
        'form_email': '884831651 @ qq.com',
        'form_password': 'BuU5whTb5P58mZx',
        'captcha_solution': captch_text,
        'captcha_id': captcha_id
    }
    print(captcha_id)
    print(captch_text)
    response = requests.post(url=url, data=formData, headers=headers)
    print('zwp2019' in response.text)


if __name__ == '__main__':
    url = 'https://www.douban.com/accounts/login'
    spider(url)
