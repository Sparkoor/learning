"""

"""


# import re
#
# reg = re.compile("[\'\[\]]+")
# reg_time = re.compile("\d{4}[-]\d{1,2}[-]\d{1,2}")
# # 居然有两千多个作者
# file = r"D:\workspace\pproject\NMF\analysisData\author.txt"
# with open(file, 'r') as f:
#     for i in f:
#         print(type(i))
#         s = re.sub(reg, '', i)
#         s = re.sub(reg_time, '', s)
# l = s.strip().split(',')
# print(len(l))
def read(lines):
    num = 0

    with open(r"D:\workspace\pproject\NMF\analysisData\dblplitter.txt") as f:
        for line in f:
            lines.append(line)
            num += 1
            if num % 10 == 0:
                yield lines
            if num == 100:
                break


if __name__ == '__main__':
    lines = []
    for i in read(lines):
        print(i)
        lines.clear()
