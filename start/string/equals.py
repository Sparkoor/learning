class equals:
    '''朴素串匹配算法'''

    def native_matching(self, t, p):
        m, n = len(t), len(p)
        i, j = 0, 0
        while i < m and j < n:
            if t[i] == p[i]:
                i, j = i + 1, j + 1
            else:
                ##j=j-i+1 t的起始的后一个开始
                i, j = 0, j - i + 1
        if i == m:
            ##返回开始下标
            return j - i
        return -1

    '''无回溯串匹配算法'''

    def gen_pnext(self, p):
        '''生成针对p中各位置i的下一检查位置表，用于KMP算法'''
        i, k, m = 0, -1, len(p)
        pnext = [-1] * m
        while i < m - 1:
            if k == -1 or p[i] == p[k]:
                i, k = i + 1, k + 1
                if p[i] == p[k]:
                    pnext[i] = pnext[k]
                else:
                    pnext[i] = k
            else:
                k = pnext[k]
        return pnext

    def march_KMP(self, t, p, pnext):
        '''KMP串匹配，主函数'''
        j, i = 0, 0
        n, m = len(t), len(p)
        while j < n and i < m:
            if i == -1 or t[j] == p[i]:
                j, i = j + 1, i + 1
            else:
                i = pnext[i]
        if i == m:
            return j - i
        return -1

