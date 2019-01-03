"""使用set取交集，还可以取并集，合并，更新等"""


class Solution:
    """
     这是去掉重复取的交集
    """

    def intersect(self, num1, num2):
        num1_1 = set(num1)
        num2_1 = set(num2)
        return num1_1.intersection(num2_1)

    """
    重复的数也要算，那只能看排序的了
    """

    def intersect1(self, nums1, nums2):
        nums1.sort()
        nums2.sort()
        num = []
        i, j = len(nums1) - 1, len(nums2) - 1
        while i >= 0 and j >= 0:
            if nums1[i] == nums2[j]:
                num.append(nums1[i])
                i = i - 1
                j = j - 1
            elif nums1[i] > nums2[j]:
                i = i - 1
            elif nums1[i] < nums2[j]:
                j = j - 1
        return num


if __name__ == '__main__':
    num1 = [1, 2, 3, 5, 1, 1]
    num2 = [2, 1, 3, 4]

    s = Solution
    n = s.intersect1(s, num1, num2)
    print(n)

    num = s.intersect(s, num1, num2)
    print(list(num))
