import numpy as np

import os, sys
import tensorflow as tf
# note：拼装参数
import argparse
# note:不明确
from functools import reduce
import math
import time

from keras.models import Model
from keras import regularizers, optimizers
from keras.layers import Input, Dense, concatenate
from keras import backend as K
from keras.models import load_model

from .dataset import read_data_sets, embedding_lookup

from commonUtils.Loggings import *

logger = Logger().getLogger()

# 声明数据
parser = argparse.ArgumentParser("hyper-network embedding", fromfile_prefix_chars='@')
# 数据集文件的路径
parser.add_argument("--data_path", type=str, help='Directory to load data.')
# 保存的文件路径
parser.add_argument("--save_path", type=str, help='Directory to save data')
# 嵌入的向量的大小 -s 是选项字符串的名字或列表
parser.add_argument("-s", "--embedding_size", type=int, nargs=3, default=[32, 32, 32],
                    help="the embedding dimension size")
# 路径的前缀
parser.add_argument('--prefix_path', type=str, default='model', help='.')
# 隐藏层的大小
parser.add_argument('--hidden_size', type=int, default=64, help='the hidden full connected layer size')
# 一次循环
parser.add_argument('-e', '--epochs_to_train', type=int, default=10,
                    help='number of epoch to train.each epoch processes the training data once complete')
# 训练一次进行训练的大小
parser.add_argument('-b', '--batch_size', type=int, default=16, help='number of training examples processed per step')
# 学习率 note：这个在哪用的
parser.add_argument('-lr', '--learning_rate', type=float, default=0.01, help='init learning rate')
# 在最终的损失函数用的，大小可以表示偏重于一阶邻近还是二阶邻近
parser.add_argument('-a', '--alpha', type=float, default=1, help='radio of autoencode loss')
# 为什么还用到了正例
parser.add_argument('-neg', '--num_neg_samples', type=int, default=5, help='neggative samples per training example')
# note：存放什么的吧
parser.add_argument('-o', '--options', type=str, help='options files to read,if empty,stdin is used')
# 随机的种子
parser.add_argument('--seed', type=int, help='random seed')


class hypergraph(object):
    """
    超图？？？
    """

    def __init__(self, options):
        self.options = options
        self.build_model()

    def aparse_autoencoder_error(self, y_true, y_pred):
        """
        二阶近邻误差
        :param y_true:
        :param y_pred:
        :return:
        """
        return K.mean(K.square(K.sign(y_true) * (y_true - y_pred)), axis=1)

    def build_model(self):
        # 模型的输入Input实例化一个keras张量。note：options是args
        self.inputs = [Input(shape=(self.options.dim_feature[i],), name='input_{}'.format(i), dtype='float') for i in
                       range(3)]
        # self.input是做什么的
        self.encodes = [
            Dense(self.options.embedding_size[i], activation='tanh', name='encode_{i}'.format(i))(self.inputs[i]) for i
            in range(3)]
        # regularizers是 正则化 我记得是
        self.decodes = [Dense(self.options.dim_feature[i], activation='sigmoid', name='decode_{}'.format(i),
                              activity_regularizer=regularizers.l2(0.0))(self.encodes[i]) for i in range(3)]
        # note:???
        self.merge = concatenate(self.encodes, axis=1)
        # note:一个括号是参数 一个括号是传入的张量
        self.hidden_layer = Dense(self.options.hidden_size, activation='tanh', name='full_connected_layer')(self.merged)
        self.output_layer = Dense(1, activation='sigmoid', name='classify_layer')(self.hidden_layer)
        # 构建模型
        self.model = Model(inputs=self.inputs, outputs=self.decodes + [self.output_layer])
        self.model.compile(optimizers=optimizers.RMSprop(lr=self.options.learming_rate),
                           loss_weights=[self.options.alpha] * 3 + [0.1], metrics=dict(
                [('decodes_{}'.format(i), 'mse') for i in range(3)] + [('classify_layer', 'accuracy')]))
        self.model.summary()

    def train(self, dataset):
        """

        :param dataset:
        :return:
        """
        # 参数都是什么
        self.hist = self.model.fit_generator(
            dataset.train.next_batch(dataset.embeddings, self.options.batch_size,
                                     num_neg_samples=self.options.num_neg_sample),
            validation_data=dataset.test.next_batch(dataset.embeddings, self.options.batch_size,
                                                    num_neg_sample=self.options.num_neg_sample),
            validation_steps=1,
            steps_per_epoch=math.ceil(dataset.train.nums_example / self.options.batch_size),
            epochs=self.options.epochs_to_train, verbose=1
        )

    def predict(self, embeddings, data):
        """
        预测
        :param embeddings:
        :param data:
        :return:
        """
        test = embedding_lookup(embeddings, data)
        return self.model.predict(test, batch_size=self.options.batch_size)[3]

    def fill_feed_dict(self, embeddings, nums_type, x, y):
        """

        :param embeddings:
        :param nums_type:
        :param x:
        :param y:
        :return:
        """
        batch_e = embedding_lookup(embeddings, x)
        return (
            dict([('input_{}'.format(i), batch_e[i]) for i in range(3)]),
            dict([('decode_{}'.format(i), batch_e[i]) for i in range(3)] + [('classify_layer', y)]))
        return res

    def get_embeddings(self, dataset):
        shift = np.append([0], np.cumsum(dataset.train.nums_type))
        embeddings = []
        for i in range(3):
            index = range(dataset.train.nums_type[i])
            batch_num = math.ceil(1.0 * len(index) / self.options.batch_size)
            ls = np.array_split(index, batch_num)
            ps = []
            for j, lss in enumerate(ls):
                embed = K.get_session().run(self.encodes[i],
                                            feed_dict={self.inputs[i]: dataset.embeddings[i][lss, :].todense()})
                ps.append(embed)
            ps = np.vstack(ps)
            embeddings.append(ps)
        return embeddings

    def save(self):
        prefix = '{}_{}'.format(self.options.prefix_path, self.options.embedding_size[0])
        prefix_path = os.path.join(self.options.save_path, prefix)
        # 创建文件
        if not os.path.exists(prefix_path):
            os.makedirs(prefix_path)
        self.model.save(os.path.join(prefix_path, 'model.h5'))
        with open(os.path.join(prefix_path, 'config.txt'), 'w') as f:
            # note:options
            for key, value in vars(self.options).items():
                if value is None:
                    continue
                if type(value) == list:
                    s_v = " ".join(list(map(str, value)))
                else:
                    s_v = str(value)
                f.write(key + " " + s_v + '\n')

    def save_embeddings(self, dataset, file_name='embedding.npy'):
        """

        :param dataset:
        :param file_name:
        :return:
        """
        emds = self.get_embeddings(dataset)
        prefix = '{}_{}'.format(self.options.prefix_path, self.options.embedding_size[0])
        prefix_path = os.path.join(self.options.save_path, prefix)
        if not os.path.exists(prefix_path):
            os.makedirs(prefix_path)
        np.save(open(os.path.join(prefix_path, file_name), 'wb'), emds)

    def load(self):
        prefix_path = os.path.join(self.options.prefix_path,
                                   '{}_{}'.format(self.options.prefix_path, self.options.embedding_size[0]))
        self.model = load_model(os.path.join(prefix_path, 'model.h5'),
                                custom_objects={'sparse_autoencoder_error': self.aparse_autoencoder_error})


def load_config(config_file):
    """
    加载配置文件
    :param config_file:
    :return:
    """
    with open(config_file, 'r') as f:
        args = parser.parse_args(reduce(lambda a, b: a + b, map(lambda x: ('--' + x).strip().split(), f.readlines())))
    logger.warning("从配置文件种加载数据配置{}".format(args))
    return args


def load_hypergraph(data_path):
    args = load_config(os.path.join(data_path, 'config.txt'))
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    K.set_session(tf.Session(config=config))
    h = hypergraph(args)
    h.load()
    return h


if __name__ == '__main__':
    args = parser.parse_args()
    if args.options is not None:
        args = load_config(args.options)
    if args.seed is not None:
        np.random.seed(args.seed)
    dataset = read_data_sets(args.data_path)
    args.dim_feature = [sum(dataset.train.nums_type) - n for n in dataset.train.nums_type]
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    K.set_session(tf.Session(config=config))
    h = hypergraph(args)
    begin = time.time()
    h.train(dataset)
    end = time.time()
    logger.warning("训练结束")
    h.save()
    h.save_embeddings(dataset)
    K.clear_session()
