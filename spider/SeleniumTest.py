"""
selenium库的使用，配合无界面浏览器 headless chrome
安装chromedriver 来驱动chrome浏览器，注意版本对应
"""
from selenium import webdriver
import time


def spider():
    driver = webdriver.Chrome('D:/lib/chromedriver/chromedriver.exe')
    driver.get('https://www.hao123.com/')
    # 获取最初页面的句柄页-----------------------------------------------
    current_win = driver.current_window_handle
    # ---------------------------------------------------------------------
    # //*[@id="kw"] 找到收缩框
    search_box = driver.find_element_by_xpath('//*[@id="search-input"]')
    # 输入收索内容
    search_box.send_keys('链家')
    # 提交收索内容 提交按钮 //*[@id="su"]
    time.sleep(2)
    submit = driver.find_element_by_xpath('//*[@id="search-form"]/div[2]/input')
    submit.click()

    # 跳转完成后，获取所有窗口的句柄-----------------------------------------------
    handles = driver.window_handles
    for i in handles:
        if current_win == i:
            continue
        else:
            driver.switch_to.window(i)
    # --------------------------------------------------------------------
    time.sleep(5)
    driver.find_element_by_xpath('//*[@id="1"]/h3/a[1]').click()
    text = driver.find_element_by_xpath('//*[@id="keyword-box"]')
    text.send_keys('沈河区')
    time.sleep(5)
    # 这一步无法获取页面元素因为新打开了一个窗口，所以需要一个移动句柄页的操作//*[@id="downloads"]/ul/li[2]/a
    sublimeabout = driver.find_element_by_xpath('//*[@id="findHouse"]')
    sublimeabout.click()
    time.sleep(5)
    # 延时
    driver.implicitly_wait(10)


if __name__ == '__main__':
    spider()
