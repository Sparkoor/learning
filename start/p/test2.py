class Solution:
    def longestPalindrome(self, s):
        """
        :type s: str
        :rtype: str
        """
        strs = []
        length = len(s)
        # 使用range不用减1
        for i in range(0, length):
            temp = s[i]
            for j in range(i + 1, length):
                temp = temp + s[j]
                if self.isPhraseString(self, temp):
                    strs.append(temp)
        max = 0
        result = ''
        for m in strs:
            l = len(m)
            if max < l:
                max = l
                result = m
            elif max == l:
                result = m
        return result

    def isPhraseString(self, str):
        height = len(str)
        if height == 1:
            return True
        for i in range(0, height):
            if i <= height - i - 1 and str[i] == str[height - i - 1]:
                pass
            elif i > height - i - 1:
                pass
            else:
                return False
        return True


s = Solution
print(s.longestPalindrome(s, "babad"))
