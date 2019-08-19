class Solution(object):
    def lengthOfLongestSubstring(self, s):
        """
        无重复的最长字符串
        :type s: str
        :rtype: int
        """
        maxlen = 0
        substr = []
        for i in s:
            if i not in substr:
                print(i)
                substr.append(i)
                l = len(substr)
                if maxlen < l:
                    maxlen = l
                print(substr)
            else:
                # 需要快速定位
                index = substr.index(i) + 1
                substr = substr[index:]
                substr.append(i)
                print("restart:", substr)
        return maxlen

    def longestCommonPrefix(self, strs):
        """
        输出最长公共前串,todo:重新开始，思路错了
        :param strs:
        :return:
        """
        if len(strs) == 0:
            return ""
        first = strs[0]
        if len(first) == 0:
            return ""
        index = 0
        flag = False
        for i, v in enumerate(first):
            index = i
            for s in strs[1:]:
                if len(s) != 0 and ((len(s) - 1) < i or s[i] != v):
                    flag = True
                    break
            if flag:
                break
        if index == 0:
            return first[0]
        elif index == 0 and flag:
            return ""
        else:
            return first[:index]


if __name__ == '__main__':
    s = Solution()
    # st = "aabcdbefghjkss"
    # max = s.lengthOfLongestSubstring(st)
    # print(max)
    # l = ['q', 'd', 'd', 'b']
    # i = l.index('d')
    # print(i)
    c = 'dssacddd'
    d = 'asassc'
    m = 'asaddddjidjijis'
    f = 'asa'
    strs = [""]
    m = s.longestCommonPrefix(strs)
    print(m)
    print(len(""))
