class Solution:
    '''判断是否是回文字符串，只考虑字符和数字'''

    def isPalindrome(self, s):
        """
        :type s: str
        :rtype: bool
        """
        if len(s) == 1:
            return True
        i, j = 0, len(s) - 1
        while i < j:
            a, b = s[i], s[j]
            while not self.isParseOrNumber(a) and i < j:
                i += 1
                a = s[i]
            while not self.isParseOrNumber(b) and i < j:
                j -= 1
                b = s[j]
            if s[i] != s[j]:
                return False
            else:
                i += 1
                j -= 1
        return True

    def isParseOrNumber(self, s):
        if str(s).isalnum() or str(s).isalpha():
            return True
        else:
            return False

    '''回文串结束'''


t = Solution
s = ','
if t.isParseOrNumber(t, s):
    print("是数字")
else:
    print("不是")
