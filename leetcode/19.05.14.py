"""
字母异位词
"""


class Solution:
    def isAnagram(self, s: str, t: str) -> bool:
        """

        :param s:
        :param t:
        :return:
        """
        m = len(s)
        n = len(t)
        if n != m:
            return False

