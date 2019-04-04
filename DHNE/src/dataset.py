"""
准备数据集
"""
import numpy as np
import os
import copy
import collections
import scipy.io as sio
import operator
from scipy.sparse import csr_matrix
from scipy.sparse import vstack as s_vstack
import sys
import itertools
from commonUtils.Loggings import Logger

logger = Logger().getLogger()

# note:作用效果产生一个带有名称的元组
Datasets = collections.namedtuple('Datasets',
                                  ['train', 'test', 'embedding', 'node_cluster', 'labels', 'idx_label', 'label_name'])


class DataSet(object):
    def __init__(self, edge, nums_type, **kwargs):
        self.edge = edge
        # edge是什么类型，转化成不可变的
        self.edge_set = set(map(tuple, edge))
        self.nums_type = nums_type
        self.kwargs = kwargs
        self.nums_examples = len(edge)
        self.epochs_complete = 0
        self.index_in_epoch = 0

    def next_batch(self, embedding, batch_size=16, num_neq_sample=1, pair_radio=0.9, sparse_input=True):
        """
        Return the next 'batch_size' example from this data set.
        if num_neg_sample=0,there is no negative sampling.
        :param embedding: todo:????
        :param batch_size:
        :param num_neq_sample:
        :param pair_radio: 成对的概率吗
        :param sparse_input:todo:?????
        :return:
        """
        while 1:
            start = self.index_in_epoch
            self.index_in_epoch += batch_size
            if self.index_in_epoch > self.nums_examples:
                self.epochs_complete += 1
                np.random.shuffle(self.edge)
                start = 0
                logger.info("完成一个迭代")
                assert self.index_in_epoch <= self.nums_examples
            end = self.index_in_epoch
            # note:作用不明
            neg_data = []
            for i in range(start, end):
                # note：
                n_neg = 0
                while (n_neg < num_neq_sample):
                    # 深度copy的作用，  这是超边的起点
                    index = copy.deepcopy(self.edge[i])
                    mode = np.random.randint()
                    if mode < pair_radio:
                        type_ = np.random.randint(3)
                        # 随机选择节点类型
                        node = np.random.randint(self.nums_type[type_])
                        # note：数据结构不明确
                        index[type_] = node
                    else:
                        # note:不明确
                        types_ = np.random.choice(3, 2, replace=False)
                        node_1 = np.random.randint(self.nums_type[types_[0]])
                        node_2 = np.random.randint(self.nums_type[types_[1]])
                        index[types_[0]] = node_1
                        index[types_[1]] = node_2
                    if tuple(index) in self.edge_set:
                        continue
                    n_neg += 1
                    # 添加一条超边
                    neg_data.append(index)
            if len(neg_data) > 0:
                # note:np.vstack是把列表元组转化成numpy数组
                batch_data = np.vstack((self.edge[start:end], neg_data))
                nums_batch = len(neg_data)
                labels = np.zeros(nums_batch)
                labels[0:end - start] = 1
                # note：permutation的作用随机排列一个数列
                perm = np.random.permutation(nums_batch)
                # 打乱顺序吗
                batch_data = batch_data[perm]
                labels = labels[perm]
            else:
                # 另一种初始化
                batch_data = self.edge_set[start:end]
                nums_batch = len(batch_data)
                labels = np.ones(len(batch_data))
            batch_e = embedding_lookup(embedding, batch_data, sparse_input)
            yield (dict([('input_{}'.format(i), batch_e[i]) for i in range(3)]),
                   dict([('decode_{}'.format(i), batch_e[i]) for i in range(3)] + [('classify', labels)]))


def embedding_lookup(embedding, index, sparse_input=True):
    """
    note：嵌入什么？？ 嵌入成向量
    :param embedding:
    :param index:
    :param sparse_input:
    :return:
    """
    if sparse_input:
        #
        return [embedding[i][index[:, i], :].todense() for i in range(3)]
    else:
        return [embedding[i][index[:, i], i] for i in range(3)]


def read_data_sets(train_dir):
    """

    :param train_dir:
    :return:
    """
    # note:npz是numpy读取的文件类型
    TRAIN_FILE = 'train_data.npz'
    TEST_FILE = 'test_data.npz'
    # 连接路径
    data = np.load(os.path.join(train_dir, TRAIN_FILE))
    # note:这个DataSet是类吗
    train_data = DataSet(data['train_data', data['nums_type']])
    labels = data['labels'] if 'idx_label' in data else None
    idx_label = data['idx_label'] if 'label_name' in data else None
    label_set = data['label_name'] if 'label_name' in data else None
    # 删除数据
    del data
    # 加载数据
    data = np.load(os.path.join(train_dir, TEST_FILE))
    test_data = DataSet(data['test_data'], data['nums_type'])
    node_cluster = data['node_cluster'] if 'node_cluster' in data else None
    test_labels = data['labels'] if 'labels' in data else None
    del data
    # todo:这是哪一步操作
    embeddings = generate_embeddings(train_data.edge, test_data.nums_type)
    # 产生一个带名称的元组
    return Datasets(train=train_data, test=test_data, embedding=embeddings, node_cluster=node_cluster, labels=labels,
                    idx_label=idx_label, label_name=label_set)


def generate_H(edge, nums_type):
    """
    产生矩阵H
    :param edge:
    :param nums_type:
    :return:
    """
    nums_examples = len(edge)
    # 因为只有三个类型 a[row_ind[k], col_ind[k]] = data[k]？？？note：不是很明白，是按类型分的,它的数据长度是按顺序分配的吗，如果是就可以理解了
    H = [csr_matrix((np.ones(nums_examples), (edge[:, i], range(nums_examples))), shape=(nums_type[i], nums_examples))
         for i in range(3)]
    return H


def dense_to_onehot(labels):
    """
    todo：还不知到是做什么的
    :param labels:
    :return:
    """
    return np.array(map(lambda x: [x * 0.5 + 0.5, x * -0.5 + 0.5], list(labels)), dtype=float)


def generate_embeddings(edge, nums_type, H=None):
    """
    生成嵌入
    :param edge:
    :param nums_type:
    :param note；H:
    :return: embeddings 二阶近邻吗
    """
    if H is None:
        H = generate_H(edge, nums_type)
    # note：为什么同种类型的不能用，这是求二阶邻近的，计算两点存在边，
    embeddings = [H[i].dot(s_vstack([H[j] for j in range(3) if j != i]).T).astype('float') for i in range(3)]
    for i in range(3):
        col_max = np.array(embeddings[i].max(0).todense()).flatten()
        _, col_index = embeddings[i].nonzero()
        # note:这是做什么的
        embeddings[i].data /= col_max[col_index]
    return embeddings
