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

parser = argparse.ArgumentParser("hyper-network embedding", fromfile_prefix_chars='@')
parser.add_argument("--data_path", type=str, help='Directory to load data.')
parser.add_argument("--save_path", type=str, help='Directory to save data')
parser.add_argument("-s", "--embedding_size", type=int, nargs=3, default=[32, 32, 32],
                    help="the embedding dimension size")
parser.add_argument('--prefix_path', type=str, default='model', help='.')
parser.add_argument('--hidden_size', type=int, default=64, help='the hidden full connected layer size')
parser.add_argument('-e', '--epochs_to_train', type=int, default=10,
                    help='number of epoch to train.each epoch processes the training data once complete')
parser.add_argument('-b', '--batch_size', type=int, default=16, help='number of training examples processed per step')
parser.add_argument('-lr', '--learning_rate', type=float, default=0.01, help='init learning rate')
parser.add_argument('-a', '--alpha', type=float, default=1, help='radio of autoencode loss')
parser.add_argument('-neg', '--num_neg_samples', type=int, default=5, help='neggative samples per training example')
parser.add_argument('-o', '--options', type=str, help='options files to read,if empty,stdin is used')
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
    def predict(self,embeddings,data):
        """
        预测
        :param embeddings:
        :param data:
        :return:
        """
