"""

"""

if __name__ == '__main__':
    s = {}
    a = [12, 43, 84, 55, 72, 81, 10]
    for k, v in enumerate(a):
        s[k] = v
    # 后面是设置默认值的
    print(s.get(8, 10000))
