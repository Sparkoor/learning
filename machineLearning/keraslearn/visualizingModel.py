"""
可视化模型
将特征图可视化的正确方法是将每个通道的内容分别绘制成二位图像
"""
# 加载用来保存的训练模型
from keras.models import load_model
#
# model = load_model('cat.h5')
# # 作为提醒
# model.summary()

img_path = r'D:\workspace\pproject\machineLearning\keraslearn\sample\cat.jpeg'

# 图像预处理为一个4D张量
from keras.preprocessing import image
import numpy as np
import matplotlib.pyplot as plt

img = image.load_img(img_path, target_size=(150, 150))
img_tensor = image.img_to_array(img)
# 增加一维吧
img_tensor = np.expand_dims(img_tensor, axis=0)
# 训练模型的输入都用这种方法预处理
img_tensor /= 255
# 其形状为(1,150,150,3)
print(img_tensor.shape)

plt.imshow(img_tensor[0])
plt.show()

lay_out