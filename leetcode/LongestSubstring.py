from commonUtils.Loggings import *

logger = Logger().getLogger()


class Solution:
    def lengthOfLongestSubstring(self, s: str) -> int:
        """
        这是个新样式吗，求最长无重复子序列
        :param s:
        :return:
        """
        sList = list(s)
        maxSize = 0
        sSize = len(sList)
        sSet = set(sList)
        # 如果整个字符串为最长的
        if len(sSet) == sSize:
            return sSize
        for i in range(0, sSize):
            for j in range(i, sSize):
                sm = sList[i:j + 1]
                subsSize = len(sm)
                if len(set(sm)) == subsSize:
                    if maxSize < subsSize:
                        maxSize = subsSize
        return maxSize

    def lengthOfLongestSubstringMain(self, s: list) -> int:
        sList = list(s)
        return self.lengthOfLongestSubstring2(sList)

    def lengthOfLongestSubstring2(self, sList: list) -> int:
        """
        采用双指针的模式
        :param s:
        :return:
        """

        if len(sList) == len(set(sList)):
            return len(sList)
        j = len(sList) - 1
        oldJ = j
        i = 0
        oldI = i
        while True:
            if sList[i] in sList[i + 1: j]:
                i += 1
            if sList[j] in sList[i:j]:
                j -= 1
            subList = sList[i:j + 1]
            if len(subList) == len(set(subList)):
                return len(subList)
            else:
                if oldI == i and oldJ == j:
                    llong = self.lengthOfLongestSubstring2(sList[i:j])
                    rlong = self.lengthOfLongestSubstring2(sList[i + 1: j + 1])
                    return max(llong, rlong)
                else:
                    oldJ = j
                    oldI = i
                    continue

    def lengthOfLongestSubstring3(self, s: str, p='first') -> int:
        """
        使用定位,每次遇见具有重复字符的串就将其分成两部分
        :param sList:
        :return:
        """
        logger.info("{}:{}".format(p, s))
        sLen = len(s)
        sSetLen = len(set(s))
        if sLen == sSetLen or sSetLen < 2:
            return sLen
        i = 0
        j = sLen - 1
        while True:
            t = s[i]
            s2 = s[i + 1:sLen]
            maxL = 0
            if t in s2:
                j = s2.index(t)
                # 左边部分
                logger.critical(s[i: j])
                left = self.lengthOfLongestSubstring3(s[i: j + 1], 'left')
                logger.error(left)
                # 边部分
                logger.critical(s2[0: j + 1])
                right = self.lengthOfLongestSubstring3(s2[0: j + 1], 'right')
                logger.warning(right)
                maxL = max(left, right)
                # todo：再设置一个出口
                logger.critical(s2[j:])
            else:
                print("ss")
            mid = self.lengthOfLongestSubstring3(s2[j:], 'mid')
            return max(maxL, mid)


if __name__ == '__main__':
    cc = Solution()
    # note:字符串可以用切片
    s = "tmmzzzuxt"
    lens = cc.lengthOfLongestSubstring3(s)
    print(lens)
    # t = s[0]
    # s2 = s[1:len(s)]
    # print(s2)
    # if t in s2:
    #     print("aaa")
    # print(s[0:2])
    # print(s.index('t'))
    # print('t' in s)
    # print(len(s))
    # print(len(set(s)))
    # a = list('asmdmdmd')
    # print(a[2])
    # print(a[0:2])
