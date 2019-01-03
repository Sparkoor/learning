class Solution:
    def plusOne(self, digits):
        """
        :type digits: List[int]
        :rtype: List[int]
        """
        length = len(digits) - 1
        flag = True
        while flag and length >= 0:
            num = digits[length]
            if (num + 1) > 9:
                digits[length] = num - 9
                if length == 0:
                    digits.insert(0, 1)
            else:
                digits[length] = num + 1
                flag = False
            length = length - 1
        return digits


"""
 字符串和数字之间的转化
"""


def plusOne(self, digits):
    """
    :type digits: List[int]
    :rtype: List[int]
    """
    self.digits = digits
    nums = int("".join([str(i) for i in digits])) + 1
    return [int(i) for i in list(str(nums))]


if __name__ == '__main__':
    digits = [1, 2, 3]
    s = Solution
    num = s.plusOne(s, digits)
    print(num)
