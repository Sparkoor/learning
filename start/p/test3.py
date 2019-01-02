class Solution:
    def isNumber(self, num):
        if num > 0 or num < 9:
            return True
        else:
            return False

    def removeDuplicates(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """

    ##快慢步数
    def moveZeroes(self, nums):
        """
        :type nums: List[int]
        :rtype: void Do not return anything, modify nums in-place instead.
        """
        n = len(nums)
        i = -1
        j = 0
        while j < n - 1:
            if nums[j] != 0:
                i += 1
                nums[i] = nums[j]
            j += 1
        for m in range(i + 1, n):
            nums[m] = 0

    ##移除元素
    def removeElement(self, nums, val):
        """
        :type nums: List[int]
        :type val: int
        :rtype: int
        """
        n = len(nums)
        i = 0
        while i < n:
            if val == nums[i]:
                nums.remove(val)
                n -= 1
            else:
                i += 1
        return len(nums)

    def removeDuplicates(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        n = len(nums)
        i = 0
        j = 1
        while j < n:
            if nums[i] == nums[j]:
                nums.remove(nums[j])
                n -= 1
            else:
                i += 1
                j = i + 1
        return len(nums)

    def removeDuplicates2(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        ##最多两个相同元素
        n = len(nums)
        i, j = 0, 1
        flag = 0
        while j < n:
            if flag < 2 and nums[i] == nums[j]:
                flag = 2
                i += 1
                j = i + 1
            elif flag == 2 and nums[i] == nums[j]:
                nums.remove(nums[j])
                n -= 1
            else:
                flag = 0
                i += 1
                j = i + 1
        return len(nums)
