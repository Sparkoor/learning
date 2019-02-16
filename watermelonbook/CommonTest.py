import numpy as np

if np.unique(['青绿' '蜷缩' '浊响' '清晰' '凹陷' '硬滑' '是'], ['是' '否']):
    print("zai")
else:
    print('fou')

a = np.arange(9).reshape(3, 3)
b = a[:, 1:]
print(b)
