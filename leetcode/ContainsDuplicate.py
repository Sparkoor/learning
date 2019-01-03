"""
判断有无重复数字，
先排序再判断
"""


class Solution:
    def containsDuplicate(self, nums):
        """
        :type nums: List[int]
        :rtype: bool
        """
        nums.sort()
        length = len(nums)
        if length == 0:
            return False
        if length == 2 and nums[0] == nums[1]:
            return True
        for i in range(1, length):
            if nums[i - 1] == nums[i]:
                return True
        return False

    def containsDuplicate2(self, nums):
        """
        :type nums: List[int]
        :rtype: bool
        """
        nums.sort()
        length = len(nums) - 1
        while length > 0:
            if nums[length] == nums[length - 1]:
                return True
            length = length - 1
        return False


if __name__ == '__main__':
    s = Solution
    nums = [3, 3]
    if s.containsDuplicate2(s, nums):
        print("dui")
    else:
        print("budui")
