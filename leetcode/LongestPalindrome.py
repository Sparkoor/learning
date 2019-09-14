class Solution:
    def longestPalindrome(self, str1, str2):
        """
        这是求最长公共子串
        :param str:
        :return: rstr
        """
        ne = self.maxCommonLength(str2)
        n, m = len(str1), len(str2)
        i, j = 0, 0
        # str1 = 'ADACADDDDAVV'
        # str2 = 'ADD'
        while i < n and j < m:
            if j == -1 or str1[i] == str2[j]:
                i += 1
                j += 1
            else:
                j = ne[j]
        if j == m:
            return i - j
        else:
            return None

    def maxCommonLength(self, str):
        """
        返回一个数组，数组包括每段字符串的最大前缀公共串的长度m
        没有m时就置变量为-1
        :param str:
        :return:
        """
        l = len(str)
        idx = list()
        i, j = 1, 0
        m = 0
        # 第一个前缀是零个
        idx.append(-1)
        while i < l:
            if str[i] == str[j]:
                i += 1
                j += 1
                m += 1
                idx.append(m - 1)
            elif str[i] != str[j]:
                idx.append(m)
                m = 0
                i += 1
                j = 0
        print('aaa')
        return idx

    def longestPalindrome2(self, s):
        """
        最长回文串
        :param s:
        :return:
        """
        length = len(s)
        if length == 1 or length == 0:
            return s
        # 暴力求解

        for i in range(length):
            while True:
                pass



if __name__ == '__main__':
    print(len(""))
