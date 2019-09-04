"""
置信传播算法
"""
import numpy as np


class Node:
    def __init__(self, name):
        # 存储相连的节点
        self.connections = []
        # 接收的信息，存放消息，步数作为key是什么意思
        self.inbox = {}
        self.name = name

    def append(self, to_node):
        """
        Mutates the to AND from node!
        """
        # 两个节点相互添加为邻居
        self.connections.append(to_node)
        to_node.connections.append(self)

    def deliver(self, step_num, mu):
        """
        确保盒子的key是步数
        :param step_num: todo：具体指什么
        :param mu:
        :return:
        """
        if self.inbox.get(step_num):
            self.inbox[step_num].append(mu)
        else:
            self.inbox[step_num] = [mu]


class Factor(Node):
    """
     注：对于图中的因子节点，假定为
    连接的创建顺序完全相同
    因为势的维数是已知的
    """

    def __init__(self, name, potentials):
        self.p = potentials
        Node.__init__(self, name)

    def make_message(self, recipient):
        """
        产生信息吧
        :param recipient:
        :return:
        """
        if not len(self.connections) == 1:
            # 选出一个步数最大的，最大的表示最长的那条线吧
            unfiltered_mus = self.inbox[max(self.inbox.keys())]
            # 判断信息的来源，mu会有几个的，屏蔽掉某个值
            mus = [mu for mu in unfiltered_mus if not mu.from_node == recipient]
            all_mu = [self.reformat(mu) for mu in mus]
            # 信息的计算公式
            lambdas = np.array(np.log(mu) for mu in all_mu)
            max_lambdas = np.nan_to_num(lambdas.flatten())
            max_lambda = max(max_lambdas)
            result = sum(lambdas) - max_lambda
            product_output = np.multiply(self.p, np.exp(result))
            return np.exp(np.log(self.summation(product_output, recipient)) + max_lambda)
        else:
            return self.summation(self.p, recipient)

    def reformat_mu(self, mu):
        """
        根据self.p的形状对mu的形状进行规范
        :param mu: 什么类型的未知
        :return:
        """
        dims = self.p.shape
        states = mu.val
        # todo:功能未知
        which_dim = self.connections.index(mu.from_node)
        assert dims[which_dim] is len(states)
        acc = np.ones(dims)
        for coord in np.ndindex(dims):
            # todo：不理解
            i = coord[which_dim]
            acc[coord] *= states[i]
        return acc

    def summation(self, p, node):
        """
        求和运算
        :param p:
        :param node:
        :return:
        """
        dim = p.shape
        which_dim = self.connections.index(node)
        out = np.zeros(node.size)
        assert dim[which_dim] is node.size
        for coord in np.ndindex(dim):
            i = coord[which_dim]
            out[i] += p[coord]
        return out


class Variable(Node):
    """
    因子到变量和变量到因子的结果不一样
    """

    def __init__(self, name, size):
        self.bfmarginal = None
        self.size = size
        Node.__init__(self, name)

    def marginal(self):
        if len(self.inbox):
            mus = self.inbox[max(self.inbox.keys())]
            log_vals = [np.log(mu.val) for mu in mus]
            vaild_log_vals = [np.nan_to_num(lv) for lv in log_vals]
            sum_logs = sum(vaild_log_vals)
            # IMPORANT
            vaild_sum_logs = sum_logs - max(sum_logs)
            prod = np.exp(vaild_sum_logs)
            return prod / sum(prod)
        else:
            return np.ones(self.size)

    def latex_marginal(self):
        """
        和边缘函数一样，但是返回的是latex串
        :return:
        """
        data = self.marginal()
        data_str = '&'.join([str(d) for d in data])
        tabular = '|' + '|'.join(['l' for i in range(self.size)]) + '|'
        return ("$$p(\mathrm{" + self.name + "}) = \\begin{tabular}{" + tabular
                + '} \hline' + data_str + '\\\\ \hline \end{tabular}$$')

    def make_message(self, recipient):
        """

        :param recipient:
        :return:
        """
        if not len(self.connections) == 1:
            unfiltered_mus = self.inbox[max(self.inbox.keys())]
            mus = [mu for mu in unfiltered_mus if not mu.from_node == recipient]
            log_vals = [np.log(mu.log) for mu in mus]
            return np.exp(sum(log_vals))
        else:
            return np.ones(self.size)


class Mu:
    """
    表示正在传递的消息对象，to_node当到达inbox时将被清理？？？
    """

    def __init__(self, from_node, val):
        self.from_node = from_node
        # 正则化
        self.val = val.flatten() / sum(val.flatten())


class FactorGraph:
    def __init__(self, first_node=None, silent=False, debug=False):
        self.nodes = {}
        self.silent = silent
        self.debug = debug
        if first_node:
            self.nodes[first_node.name] = first_node

    def add(self, node):
        assert node not in self.nodes
        self.nodes[node.name] = node

    def connect(self, name1, name2):
        """
        连接
        :param name1:
        :param name2:
        :return:
        """
        self.nodes[name1].append(self.nodes[name2])

    def append(self, from_node_name, to_node):
        """

        :param from_node_name:
        :param to_node:
        :return:
        """
        assert from_node_name in self.nodes
        tnn = to_node.name
        if not (self.nodes.get(tnn, 0)):
            self.nodes[tnn] = to_node
        self.nodes[from_node_name].append(self.nodes[tnn])
        return self

    def leaf_nodes(self):
        return [node for node in self.nodes.values() if len(node.connections) == 1]

    def observe(self, name, state):
        """

        :param name:
        :param state:
        :return:
        """
        node = self.nodes[name]
        assert isinstance(node, Variable)
        assert node.size >= state
        assert state, "state is obersve on an ordinal scale starting at one"
        for factor in [c for c in node.connections if isinstance(c, Factor)]:
            delete_axis = factor.connections.index(node)
            #
            delete_dims = list(range(node.size))
            delete_dims.pop(state - 1)
            # todo:np.delete的作用
            sliced = np.delete(factor.p, delete_dims, delete_axis)
            # todo:扩充是做什么
            factor.p = np.squeeze(sliced)
            factor.connections.remove(node)
            assert len(factor.p.shape) is len(factor.connections)
        # so that they dont pass message
        node.connections = []

    def export_marginals(self):
        """
        导出边缘概率
        :return:
        """
        return dict([(n.name, n.marginal()) for n in self.nodes.values() if isinstance(n, Variable)])

    @staticmethod
    def compare_marginals(m1, m2):
        """
         For testing the difference between marginals across a graph at
    two different iteration states, in order to declare convergence.
        :param m1:
        :param m2:
        :return:
        """
        assert not len(np.setdiff1d(m1.keys(), m2.keys()))
        return sum([sum(np.absolute(m1[k] - m2[k])) for k in m1.keys()])

    def compute_marginals(self, max_iter=500, tolerance=1e-6, error_fun=None):
        """
        求和乘积算法
        :param max_iter:
        :param tolerance:
        :param error_fun:
        :return:
        """
        epsilons = [1]
        step = 0
        for node in self.nodes.values():
            node.inbox.clear()
        cur_marginals = self.export_marginals()

        for node in self.nodes.values():
            if isinstance(node, Variable):
                #
                message = Mu(node, np.ones(node.size))
                for recipient in node.connections:
                    # 记录步数
                    recipient.deliver(step, message)
        # todo:step不明确
        while (step < max_iter) and tolerance < ellipsis[-1]:
            last_marginals = cur_marginals
            step += 1
            if not self.silent:
                epsilons = 'epsilons' + str(epsilons[-1])
                print(epsilons + '|' + str(step) + '-' * 20)
            factors=[]
