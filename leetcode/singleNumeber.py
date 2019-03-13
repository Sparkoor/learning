class Solution(object):
    """
    返回只有不重复的数字
    """

    def singleNumber(self, nums: list) -> int:
        """

        :param nums:
        :return:
        """
        l = len(nums)
        if l == 1:
            return nums[0]
        for i in range(l):
            if nums[i] in (nums[0: i] + nums[i + 1: l]):
                continue
            else:
                return nums[i]


if __name__ == '__main__':
    a = [1, 2, 3, 4, 5]
    if 1 in a:
        print("zai")
    print(a[1:3] + a[0:1])
    print(1 ^ 2)
    print(2 ^ 3)
    print(2 ^ 2)
