"""
from
https://github.com/tensorflow/models/blob/master/tutorials/image/cifar10/cifar10_input.py
"""
import tensorflow as tf

# 没有导入这个包
import tensorflow_datasets as tfds

IMAGE_SIZE = 24
# 数据集的全局描述
NUM_CLASSES = 10
NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = 50000
NUM_EXAMPLES_PER_EPOCH_FOR_EVAL = 10000


def _get_images_labels(batch_size, split, distords=False):
    """
    返回分割好的数据集
    :param batch_size:
    :param split:
    :param distords:
    :return:
    """
    # 加载数据集
    dataset = tfds.load(name='cifar10', split=split)
    # note:新认识的用法,这句话的意思未懂
    scope = 'data_augmentation' if distords else 'input'
    with tf.name_scope(scope):
        dataset = dataset.map(DataPreprocessor(distords), num_parallel_calls=10)
    # 数据集够小全部加载到内存
    dataset = dataset.prefetch(-1)
    dataset = dataset.repeat().batch(batch_size)
    iterator = dataset.make_one_shot_iterator()
    images_labels = iterator.get_next()
    images, labels = images_labels['input'], images_labels['target']
    tf.summary.image('images', images)
    return images, labels


class DataPreprocessor(object):
    """用于数据集的转换"""

    def __init__(self, disorder):
        self.disorder = disorder

    def __call__(self, record):
        """
        note；这个函数实例化就会调用
        为训练或评估处理image
        :param record:
        :return:
        """
        img = record['image']
        img = tf.cast(img, tf.float32)
        # 无序的是图片内容
        if self.disorder:
            # 随机裁剪图像的[高度、宽度]部分
            img = tf.random_crop(img, [IMAGE_SIZE, IMAGE_SIZE, 3])
            # 将图像水平随意翻转
            img = tf.image.random_flip_left_right(img)
            # todo:功能
            img = tf.image.random_brightness(img, max_delta=63)
            img = tf.image.random_contrast(img, lower=0.2, upper=1.8)
        else:
            img = tf.image.resize_image_with_crop_or_pad(img, IMAGE_SIZE, IMAGE_SIZE)
        img = tf.image.per_image_standardization(img)
        return dict(input=img, target=record['label'])


def distorted_inputs(batch_size):
    """
    构建无序输入
    :param batch_size:
    :return: 4D张量 1D张量
    """
    return _get_images_labels(batch_size, tfds.Split.TRAIN, distords=True)


def inputs(eval_data, batch_size):
    """
    构建评估数据集
    :param eval_data:
    :param batch_size:
    :return:
    """
    split = tfds.Split.TEST if eval_data == 'test' else tfds.Split.TRAIN
    return _get_images_labels(batch_size, split)
