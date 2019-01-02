class Solution:
    ##
    def sortColors(self, nums):
        """
        :type nums: List[int]
        :rtype: void Do not return anything, modify nums in-place instead.
        """
        n = len(nums)
        for i in range(0, n):
            for j in range(i + 1, n):
                if nums[i] > nums[j]:
                    temp = nums[i]
                    nums[i] = nums[j]
                    nums[j] = temp

    def findKthLargest(self, nums, k):
        """
        :type nums: List[int]
        :type k: int
        :rtype: int
        """
        nums.sort(reverse=True)
        return nums[k + 1]

    ##整合两个有序数组
    def merge(self, nums1, m, nums2, n):
        """
        :type nums1: List[int]
        :type m: int
        :type nums2: List[int]
        :type n: int
        :rtype: void Do not return anything, modify nums1 in-place instead.
        """
        if n == 0:
            return
        if m == 0:
            for i in range(0, len(nums1)):
                if i < n:
                    nums1[i] = nums2[i]
            return
        if nums1[m - 1] < nums2[0]:
            j = 0
            for i in range(m, m + n):
                nums1[i] = nums2[j]
                j += 1
        elif nums2[n - 1] < nums1[0]:
            z = m-1
            ##替换和后移没有想到同时进行的办法
            for i in range(m + n - 1, -1, -1):
                if z == n:
                    break
                nums1[i] = nums1[z]
                z -= 1
            for j in range(0, n):
                nums1[j] = nums2[j]
        else:
            tag = [0] * 2
            for t in range(0, n):
                flag = True
                for i in range(0, m):
                    tag[1] = m
                    l = m
                    if nums2[t] < nums1[i] and flag:
                        flag = False
                        tag[0] += 1
                        for j in range(l, i, -1):
                            nums1[j] = nums1[j - 1]
                        nums1[i] = nums2[t]
                        m += 1
            for i in range(tag[0], n):
                nums1[tag[1]] = nums2[i]
                tag[1] += 1


e = Solution
a = [4, 5, 6, 0, 0, 0]
b = [1, 2, 3]
e.merge(e, a, 3, b, 3)
print(a)
