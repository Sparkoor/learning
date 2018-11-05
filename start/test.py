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
            while not self.isParseOrNumber(self, a) and i < j:
                print("前面", a)
                i += 1
                a = s[i]
            while not self.isParseOrNumber(self, b) and i < j:
                print("后面", b)
                j -= 1
                b = s[j]
            if a.lower() != b.lower():
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

    '''反转元音字母'''
    def reverseVowels(self, s):
        """
        :type s: str
        :rtype: str
        """

t = Solution
s = 'A man, a plan, a canal: Panama'
# if t.isParseOrNumber(t, s):
#     print("是数字")
# else:
#     print("不是")
if t.isPalindrome(t, s):
    print("True")
else:
    print("False")
