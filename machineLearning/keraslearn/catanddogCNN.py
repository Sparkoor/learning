"""
将猫狗分类的小型卷积神经网络
"""

from keras import models
from keras import layers
from keras import optimizers
import os

def build_model():
    model = models.Sequential()
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)))
    # 最常用的就是2×2的
    model.add(layers.MaxPooling2D(2, 2))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D(2, 2))
    model.add(layers.Conv2D(128, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D(2, 2))
    model.add(layers.Conv2D(128, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D(2, 2))
    # 经过四层卷积层
    model.add(layers.Flatten())
    model.add(layers.Dense(512, activation='relu'))
    model.add(layers.Dense(1, activation='sigmoid'))

    return model


# keras中处理图像的包
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image


def normal_image(train_dir, test_dir):
    """
    图片处理
    :param train_dir:
    :param test_dir:
    :return:
    """
    # 将所有图像乘以1/255
    train_datagen = ImageDataGenerator(rescale=1. / 255)
    test_datagen = ImageDataGenerator(rescale=1. / 255)
    # 目标目录，将所有图像调整为150*150，因为使用了binary_crossentropy所以使用二进制标签
    train_generator = train_datagen.flow_from_directory(train_dir,
                                                        target_size=(150, 150)
                                                        , batch_size=20,
                                                        class_mode='binary')
    validation_generator = test_datagen.flow_from_directory(
        test_dir,
        target_size=(150, 150),
        batch_size=20,
        class_mode='binary'
    )
    return train_generator, validation_generator


def testgenera():
    train_generator, test_generator = normal_image('', '')
    for data_batch, label_batch in train_generator:
        print('data batch shape', data_batch.shape)
        print('label batch shape', label_batch.shape)
        break


def train():
    """
    使用了迭代器，迭代器的原理
    :return:
    """
    model = build_model()
    # 编译
    model.compile(loss='binary_crossentropy',
                  optimizer=optimizers.RMSprop(lr=1e-4),
                  metrics=['acc'])
    train_generator, test_generator = normal_image('', '')

    history = model.fit_generator(
        train_generator, steps_per_epoch=100,
        epochs=30,
        validation_data=test_generator,
        validation_steps=50
    )
    # 保存模型
    model.save('cat_and_dogs_small_1.h5')
    return history


import matplotlib.pyplot as plt


def plot_result():
    history = train()
    acc = history.history['acc']
    val_acc = history.history['val_acc']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs = range(1, len(acc) + 1)

    plt.plot(epochs, acc, 'bo', label='training acc')
    plt.plot(epochs, val_acc, 'b', label='Validation acc')

    plt.title('Training and validation accuracy')

    plt.legend()
    plt.figure()

    plt.plot(epochs, loss, 'bo', label='Training loss')
    plt.plot(epochs, val_loss, 'b', label='Validation')
    plt.title('Training and Validation loss')
    plt.legend()
    plt.figure()

    plt.show()


def data_strengthen():
    """
    防止数据过少出现过拟合，计算机视觉领域和在深度学习模型处理图像时几乎都会用到数据增强
    :return:
    """
    # 1、是0-180是角度值，表示图像随机旋转的角度范围。2、3表示图像在水平方向或垂直方向上平移的范围。
    # 4、是随机错切变换的角度。
    # 5、图像随机缩放的范围
    # 6、随机将一半的图像水平翻转
    # 7、full mode is
    datagen = ImageDataGenerator(
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )
    # 返回迭代器
    return datagen


def pre_Images(train_cats_dir):
    """
    显示几个随机增强后的训练图像
    :return:
    """
    fnames = [os.path.join(train_cats_dir, fname) for fname in os.listdir(train_cats_dir)]
    # 选择一张图像进行增强
    img_path = fnames[3]
    # 读取图像并调整大小
    img = image.load_img(img_path, target_size=(150, 150))
    # 将图像转换为形状为（150,150,3）的numpy数组
    x = image.img_to_array(img)
    # 将形状改变为（1,150,150,3）
    x = x.reshape((1,) + x.shape)

    datagen = data_strengthen()
    i = 0
    for batch in datagen.flow(x, batch_size=1):
        plt.figure(i)
        imgplot = plt.imshow(image.array_to_img(batch[0]))
        i += 1
        if i % 4 == 0:
            break

    plt.show()


def dropout_model():
    """
    数据增强不足以完全消除过拟合，为了进一步降低过拟合，还需要向模型中
    添加一个dropout层，添加到密集连接分类器之前
    :return:
    """
    model = models.Sequential()
    model.add(layers.Conv2D(32, (3, 3), activation='relu',
                            input_shape=(150, 150, 3)))
    model.add(layers.MaxPooling2D(2, 2))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D(2, 2))
    model.add(layers.Conv2D(128, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D(2, 2))
    model.add(layers.Conv2D(128, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D(2, 2))
    model.add(layers.Flatten())
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(512, activation='relu'))
    model.add(layers.Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer=optimizers.RMSprop(lr=1e-4),
                  metrics=['acc'])
    return model


def strength_dropout(train_dir, validation_dir):
    """
    利用数据增强生成器训练卷积神经网络
    :return:
    """
    train_datagen = ImageDataGenerator(
        rescale=1. / 255,
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
    )
    # 不能增强验证数据
    test_datagen = ImageDataGenerator(rescale=1. / 255)

    train_generator = train_datagen.flow_from_directory(
        # 文件夹
        train_dir,
        # 将所有图像调整为150*150
        target_size=(150, 150),
        #
        batch_size=32,
        # 二进制标签
        class_mode='binary'
    )
    validation_generator = test_datagen.flow_from_directory(
        validation_dir,
        target_size=(150, 150),
        batch_size=32,
        class_mode='binary'
    )
    model = dropout_model()

    history = model.fit_generator(
        train_generator,
        steps_per_epoch=100,
        epochs=100,
        validation_data=validation_generator,
        validation_steps=50
    )
    # 在接下来会用到
    model.save('cats_and_dogs_small_2.h5')
    return history


