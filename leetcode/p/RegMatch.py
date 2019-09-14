import re
"""
正则表达式测试
"""
r = re.compile("abc")
print(type(r))
l = re.search("abc", "aaabc")
print(type(l))



def match(re, text):
    def match_here(re, i, text, j):
        """检查从text[j]开始的正文是否与re[i]开始的模式匹配"""
        while True:
            if i == rlen:
                return True
            if re[i] == '$':
                return i + 1 == rlen and j == tlen
            if i + 1 < rlen and re[i + 1] == '*':
                return match_star(re[i], re, i + 2, text, j)
            if j == tlen or (re[i] != '.' and re[i] != text[j]):
                return False
            i, j = i + 1, j + 1

    def match_star(c, re, i, text, j):
        """ 在text里跳过0个或多个c后检查匹配"""
        for n in range(j, tlen):
            if match_here(re, i, text, n):
                return True
            if text[n] != c and c != '.':
                break
        return False

    rlen, tlen = len(re), len(text)
    if re[0] == '^':
        if match_here(re, 1, text, 0):
            return 0
    for n in range(tlen):
        if match_here(re, 0, text, n):
            return n
    return -1
