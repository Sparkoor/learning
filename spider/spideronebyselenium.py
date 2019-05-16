from selenium import webdriver
import time
from lxml import etree


def spider():
    driver = webdriver.Chrome('D:/lib/chromedriver/chromedriver.exe')
    # 图片异步加载还有对链接进行编码，编码规则，代码有点多
    driver.get('http://www.1kkk.com/manhua10684/')
    # 获取最初页面的句柄页-----------------------------------------------
    current_win = driver.current_window_handle
    # ---------------------------------------------------------------------
    # //*[@id="kw"] 找到收缩框//*[@id="detail-list-select-1"]/li[1]/a
    search_box = driver.find_element_by_xpath('//*[@id="detail-list-select-1"]/li[1]/a')
    search_box.click()

    # 跳转完成后，获取所有窗口的句柄-----------------------------------------------
    handles = driver.window_handles
    for i in handles:
        if current_win == i:
            continue
        else:
            driver.switch_to.window(i)
    # --------------------------------------------------------------------
    time.sleep(3)
    print(driver.page_source)
    img = driver.find_element_by_xpath('//*[@id="cp_image"]')
    print(img.text)

    # 延时
    driver.implicitly_wait(10)


if __name__ == '__main__':
    spider()
