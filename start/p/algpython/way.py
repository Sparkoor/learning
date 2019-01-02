class Solution:
    def quicksort(self, array):
        if len(array) < 2:
            return array
        else:
            pivot = array[0]
            less = [i for i in array[1:] if i < pivot]
            greater = [i for i in array[1:] if i > pivot]
            return self.quicksort(less) + [pivot] + self.quicksort(greater)




s = Solution()

arr = [1, 2, 7, 9, 0]
print(s.quicksort(arr))
