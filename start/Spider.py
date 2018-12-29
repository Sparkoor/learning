import requests
import lxml

#定制请求头，可以在浏览器中获取
headers={'user-Agent':'Mozilla/5.0 (X11; Linux x86_64\
           AppleWebKit/537.36 (KHTML, like Gecko)\
            Chrome/71.0.3578.98 Safari/537.36'}
param={'q':'a'}
#get请求 timeout重定向 params传递参数
response=requests.get("http://www.baidu.com",headers=headers,timeout=3000,params=param)
print(response.content)
print(response.encoding)
print(response.history)
print(response.status_code)

#urllib库
from urllib.request import urlopen
import urllib.request
#html为返回的二进制对象
html=urlopen("http://www.baidu.com")
#可通过read()读取
response=html.read()
#解码
print(response.decode('utf-8'))
print(html.geturl())

#通过request对象添加header头部 伪装成浏览器
request=urllib.request.Request("http://www.baidu.com",headers=headers)

#post请求 parse提供url解析的工具
from urllib import request,parse
#创建request对象
url=request.Request("http://www.baidu.com")
url.add_header('')
#对请求参数编码
post_data=parse.urlencode([('q','q')])
response=request.urlopen(url,param=post_data)

#lxml



