import math


class Solution:
    def twoSum(self, nums, target):
        for i in range(0, len(nums) - 1):
            num = target - nums[i]
            for j in range(i + 1, len(nums)):
                if num == nums[j]:
                    return [i, j]
        return None

    def findMedianSortedArrays(self, nums1, nums2):
        """
        :type nums1: List[int]
        :type nums2: List[int]
        :rtype: float
        """
        nums1.extend(nums2)
        nums1.sort()
        n = len(nums1)
        if n % 2 == 0:
            i = int(n / 2)
            print("i=" + str(i))
            return (nums1[i] + nums1[i - 1]) / 2
        else:
            i = int(n / 2)
            return nums1[i]

    # 判断是否为回文
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

    ##截取回文串
    # TODO：没想好怎么截取回文
    def longesthuiwen(self, str):
        strs = []
        length = len(str)
        print("字符串的长度", length)
        # 使用range不用减1
        for i in range(0, length):
            temp = str[i]
            for j in range(i + 1, length):
                temp = temp + str[j]
                print(temp)
                if self.isPhraseString(self, temp):
                    print("有进入", temp)
                    strs.append(temp)
        print(strs)
        max = 0
        result = ''
        for m in strs:
            l = len(m)
            if max < l:
                max = l
                result = m
        return result

    # nums = [3, 2, 4]


s = Solution
# nu = s.twoSum(None, nums, 6)
# print(nu)
# num1 = [1, 2]
# num2 = [3, 4]
# a = s.findMedianSortedArrays(None, num1, num2)
# print(a)

chuan = "babad"
print(s.longesthuiwen(s, chuan))
print("执行结束")
# if s.isPhraseString(s, "abbb"):
#     print("yes")
# else:
#     print("no")
#
# print(math.ceil(4 / 2))
