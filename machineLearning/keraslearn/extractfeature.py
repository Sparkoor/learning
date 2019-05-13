"""
特征提取
"""
from keras.applications import VGG16
from keras import models
from keras import layers
from keras import optimizers

# 1、指定模型初始化的权重检查点，2、指定模型是否包含密集连接分类器，3、输入到网络的张量形状
conv_base = VGG16(weights='imagenet', include_top=False, input_shape=(150, 150, 3))

conv_base.summary()

import os
import numpy as np
# 图像处理的生成器
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt

base_dir = ''
train_dir = os.path.join(base_dir, '')
validation_dir = os.path.join(base_dir, '')
test_dir = os.path.join(base_dir, '')

# 生成器
datagen = ImageDataGenerator(rescale=1. / 255)
batch_size = 20


def extract_feature(directory, sample_count):
    """
    使用预训练的卷积基提取特征
    :param directory:
    :param sample_count:
    :return:
    """
    # 卷积结束以后的操作
    features = np.zeros(shape=(sample_count, 4, 4, 512))
    labels = np.zeros(shape=(sample_count))
    generator = datagen.flow_from_directory(
        directory,
        target_size=(150, 150),
        batch_size=batch_size,
        class_mode='binary'
    )
    i = 0
    for input_batch, labels_batch in generator:
        # note：预测的是什么东西？？
        features_batch = conv_base.predict(input_batch)
        # 分批的切片
        features[i * batch_size:(i + 1) * batch_size] = features_batch
        labels[i * batch_size:(i + 1) * batch_size] = labels_batch
        i += 1
        if i * batch_size >= sample_count:
            break

    return features, labels


train_features, train_labels = extract_feature(train_dir, 2000)
validation_features, validation_labels = extract_feature(validation_dir, 1000)
test_features, test_labels = extract_feature(test_dir, 1000)

# 看看是什么形状
print(train_features.ndim())
# 将其展平
train_features = np.reshape(train_features, (2000, 4 * 4 * 512))
validation_features = np.reshape(validation_features, (1000, 4 * 4 * 512))
test_features = np.reshape(test_features, (1000, 4 * 4 * 512))

# 定义并训练密集连接分类器
model = models.Sequential()
model.add(layers.Dense(256, activation='relu', input_dim=4 * 4 * 512))
# 防止过拟合
model.add(layers.Dropout(0.5))
model.add(layers.Dense(1, activation='sigmoid'))

model.compile(optimizer=optimizers.RMSprop(lr=2e-5), loss='binary_crossentropy', metrics=['acc'])

history = model.fit(train_features, train_labels, epochs=30, batch_size=20,
                    validation_data=(validation_features, validation_labels))

