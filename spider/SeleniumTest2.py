from selenium import webdriver
import time


# 一些本地配置
def spider():
    driver = webdriver.Chrome('D:/lib/chromedriver/chromedriver.exe')
    driver.get('https://www.baidu.com/')
    # 获取首次页的handle
    handle = driver.current_window_handle
    drivers = driver.window_handles
    print('第一次所有的handle' + str(drivers))
    print('第一个handle' + str(handle))
    # 看看效果
    time.sleep(3)
    # 获取收索文本框
    search_box_baidu = driver.find_element_by_xpath('//*[@id="kw"]')
    # 填写需要查询的内容
    search_box_baidu.send_keys('狗子')
    # 看看效果
    time.sleep(3)
    # 获取‘百度一下’按钮
    search_button_baidu = driver.find_element_by_xpath('//*[@id="su"]')
    # 点击按钮
    search_button_baidu.click()
    # 看看效果
    time.sleep(3)
    # 找到第一条结果，点击进入
    first_result = driver.find_element_by_xpath('//*[@id="1"]/h3/a')
    first_result.click()
    # 看看效果
    time.sleep(3)
    # 切换到首页的handle
    second_handle = driver.current_window_handle
    second_drivers = driver.window_handles
    print('第二次所有的handle' + str(second_drivers))
    print('第一个handle' + str(second_handle))
    # 切换driver
    for d in second_drivers:
        if d == handle:
            pass
        else:
            driver.switch_to.window(d)
    second_result = driver.find_element_by_xpath('//*[@id="topRS"]/a[1]')
    second_result.click()
    time.sleep(3)


if __name__ == '__main__':
    spider()
