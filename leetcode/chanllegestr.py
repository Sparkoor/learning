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

    def longestCommonPrefix2(self, strs):
        """
        计算最长公共前串
        :param strs:
        :return:
        """
        min_str = float('inf')
        min_index = 0
        for i, str in enumerate(strs):
            min_temp = len(str)
            if min_temp < min_str:
                min_str = min_temp
                min_index = i
        if min_str == 0:
            return ""
        mark = strs[min_index]
        # 确保把所有的字符串都遍历才退出循环的
        flag = True
        idex = 0
        for i, v in enumerate(mark):
            for str in strs:
                if str == mark:
                    continue
                if v != str[i]:
                    idex = i - 1
                    flag = False
                    break
        if not False:
            return mark[:idex]


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
    m = s.longestCommonPrefix2(strs)
    print(m)
    print(len(""))
