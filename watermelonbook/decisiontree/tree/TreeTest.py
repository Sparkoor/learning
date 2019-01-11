tree = {'root': {'chil1': {'grandson1': 'maxiu', 'grandson2': 'liyliy'}, 'child2': 'jerry'}}
import matplotlib.pyplot as plt

"""绘制决策树函数,目前还没可以画树的阶段"""
# 定义分支节点样式
decisionNode = dict(boxstyle='sawtooth', fc='0.8')
# 定义叶节点样式
leafNode = dict(boxstle='round4', fc='0.8')
# 定义箭头标识样式
arrow_args = dict(arrowstyle='<-')


# 计算树的叶子节点数
def getLeafNum(tree):
    leafNum = 0
    firstStr = list(tree.keys())
