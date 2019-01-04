"""
selenium库的使用，配合无界面浏览器 headless chrome
安装chromedriver 来驱动chrome浏览器，注意版本对应
"""
from selenium import webdriver
import time


def spider():
    driver = webdriver.Chrome('D:/lib/chromedriver/chromedriver.exe')
    driver.get('https://www.baidu.com')
    # //*[@id="kw"] 找到收缩框
    search_box = driver.find_element_by_xpath('//*[@id="kw"]')
    # 输入收索内容
    search_box.send_keys('python')
    # 提交收索内容 提交按钮 //*[@id="su"]
    time.sleep(2)
    submit = search_box.find_element_by_xpath('//*[@id="su"]')
    submit.click()
    time.sleep(5)
    text = driver.find_element_by_xpath('//*[@id="1"]/h3/a[1]')
    print(text.text)
    text.click()
    time.sleep(5)
    # 这一步的时候速度就会慢了
    about = driver.find_element_by_xpath('//*[@id="downloads"]/ul/li[1]/a')
    about.click()
    time.sleep(5)
    # 延时
    driver.implicitly_wait(10)


if __name__ == '__main__':
    spider()
