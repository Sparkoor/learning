# -*- coding: utf-8 -*-
import scrapy
from twisted.internet import _win32stdio

# 导入定义的数据
from pabyscraoy.items import PabyscraoyItem


class BaiduSpider(scrapy.Spider):
    # 爬取的网站名称
    name = 'baidu'
    # 爬取的网站的域名，不在该域名下的网站不爬取
    allowed_domains = ['baidu.com']
    # 爬取网站的初始地址
    start_urls = ['http://baidu.com/']

    # 处理返回的数据
    def parse(self, response):
        print('=======================================================================')
        # 初始化数据容器
        item = PabyscraoyItem()
        item['title'] = response.xpath('//*[@id="u_sp"]/a[1]/text()').extract()[0]
        item['url'] = response.xpath('//*[@id="u_sp"]/a[1]/@href').extract()[0]
        print('------------------------------' + str(item['title']))
        return item
