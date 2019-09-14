"""
正则表达式学习
"""
import re

"""
. 任意字符
* 任意次数
^ 表示开头
$ 表示结尾
? 非贪婪模式，提取第一个字符
+ 至少出现一次
{1} 出现一次
{3,} 出现至少3次
{2,5} 出现2到5次
\d 匹配数字
[\u4E00-\u9FA5] 汉字匹配
| 或的关系
[] 满足任意一个都可以,[1233],区间[0-9] [^1]非1
\s 为空格 \S 非空格
\w 匹配[A-Za-z0-9_]
\W 反匹配
"""
line = r'this is python 数据预处理，这是一个测试？然后看看效果，2019年,狗狗狗2018年。there are some dogs..'
# 开头为t任意次数,不能接着匹配
reg_str1 = '^t.{6}'
# 返回的为分组的形式
res = re.match(reg_str1, line)
# if res:
#     print(res)
#     # print(res.group(1))
#     print(res.group(0))
# 任意开头 s结尾,匹配多个s
reg_str2 = '.*(s+)'
# 提取年，note:很多情况要加 .* 这样才能匹配上里面的
reg_str3 = '.*?(\d{4}年)'
# 括号表示正则的开始
# line = '张三出生于1994-12-1'
line = '张三出生于css,,[]199[]4[年12[月1日'
# note：使用| 或时也要加括号
reg_str4 = '.*出生于(\d{4}[年/-]\d{1,2}([月/-]\d{1,2}日|[月/-]$|$))'
res = re.match(reg_str4, line)
# if res:
#     print(res.group(1))

# 正则过滤掉特殊符号，标点，英文，数字等
reg_str5 = '[a-zA-Z0-9,!"#$%&\'()*+，./:：；;|<=>?\[\]^_]+'
reg_str5_1 = '[a-zA-z0-9,!"#$%&\'()*+，./;：；|<=>?、\[\]^]+'
# 去空格
reg_str6 = '\s+'
line = re.sub(reg_str5, ' ', line)
print(line)
# line = line.replace(reg_str5, " ")
# print(line)
