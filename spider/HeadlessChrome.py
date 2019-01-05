"""
不用打开浏览器爬取内容
爬异步加载的有点麻烦
"""
from selenium import webdriver


def headless(url):
    chrome_options = webdriver.ChromeOptions()
    chrome_options.add_argument('--ignore-certificate-errors')
    chrome_options.add_argument('--headless')
    chrome_options.add_argument('--disable-gpu')
    # binary_location指向浏览器安装目录
    chrome_options.binary_location = (r'C:\Program Files (x86)\Google\Chrome\Application\chrome.exe')
    driver = webdriver.Chrome(executable_path=r'D:/lib/chromedriver/chromedriver.exe', options=chrome_options)
    driver.get(url)
    search_text = driver.find_element_by_xpath('//html/body')
    print(search_text.text)


if __name__ == '__main__':
    headless('https://www.baidu.com')
