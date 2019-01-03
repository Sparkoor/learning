nums = [1, 6, 4, 3, 2]
nums1 = [1, 6]
nums.insert(0, 9)
print(nums)

for i in range(0, 5):
    print(i)

aa = set(nums)
bb = set(nums1)
cc = aa.intersection(nums1)
print(cc)
