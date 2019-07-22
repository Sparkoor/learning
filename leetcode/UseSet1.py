class Solution(object):
    def intersection(self, nums1, nums2):
        """
        :type nums1: List[int]
        :type nums2: List[int]
        :rtype: List[int]
        """
        res = set()
        for i in nums1:
            if i in nums2:
                res.add(i)
        print(list(res))
        return list(res)

    def isHappy(self, n):
        """
        暂时没有合适的办法
        :type n: int
        :rtype: bool
        """
        # 使用set判断之前的数有没有出现
        ss = set()
        while (True):
            a = str(n)
            sum = 0
            for i in a:
                sum += (int(i) * int(i))
            n = sum
            ss.add(sum)
            if sum == 1:
                print("true")
                return True
            if sum in ss:
                print("false")
                return False


if __name__ == '__main__':
    d = Solution()
    # num1 = [1, 2, 2, 1]
    # num2 = [2, 2]
    # d.intersection(num1, num2)
    d.isHappy(2)
