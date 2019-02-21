import matplotlib.pyplot as plt

# 定义节点样式
decisinNode = dict(boxstyle='sawtooth', fc='0.8')
leafNode = dict(boxstyle='round4', fc='0.8')
# 设置箭头
arrow_args = dict(arrowstyle='<-')


def plotNode(noteTxt, centerPt, parentPt, nodeType):
    """
    绘制节点
    :param noteTxt: 节点的名称
    :param centerPt: 指向的节点
    :param parentPt: 尾巴
    :param nodeType: 节点的样式
    :return: null
    """
    createPlot.ax1.annotate(noteTxt, xy=parentPt, xycoords='axes fraction', xytext=centerPt, textcoords='axes fraction',
                            va='center', ha='center', bbox=nodeType, arrowprops=arrow_args)


def plotMidText(cntrpt, parentPt, txtString):
    """
    在父子节点之间填充信息
    :param cntrpt: tuple 包含坐标信息，这是叶
    :param parentPt: 这是父
    :param txtString: 这是文本
    :return:
    """
    xMid = (parentPt[0] - cntrpt[0]) / 2 + cntrpt[0]
    yMid = (parentPt[1] - cntrpt[1]) / 2 + cntrpt[1]
    return createPlot.ax1.text(xMid, yMid, txtString)


def plotTree(myTree, parentPt, nodeTxt):
    numLeafs = getNumLeafs(myTree)
    numDepth = getDept(myTree)
    firstStr = myTree.keys[0]

def createPlot():
    """
    创建画布
    :return:
    """
    fig = plt.figure(1, facecolor='white')
    fig.clf()

    # 这是一种全局变量
    createPlot.ax1 = plt.subplot(111, frameon=False)


def getNumLeafs(myTree):
    """
    获取树的叶的数量
    :param myTree:
    :return:
    """
    numLeafs = 0
    firstStr = myTree.keys()[0]
    secondDict = myTree[firstStr]
    for key in secondDict.keys():
        if type(secondDict[key]).__name__ == 'dict':
            numLeafs += getNumLeafs(secondDict[key])
        else:
            numLeafs += 1
    return numLeafs


def getDept(myTree):
    """
    获取树的高度
    :param myTree:
    :return:
    """
    maxDepth = 0
    firstChild = myTree.keys[0]
    secondDict = myTree[firstChild]
    for key in secondDict.keys:
        if type(secondDict[key]).__name__ == 'dict':
            numDepth = 1 + getDept(secondDict[key])
        else:
            numDepth = 1
        if numDepth > maxDepth:
            maxDepth = numDepth
    return maxDepth


def retrieveTree(i):
    """
    生成一个决策树使用
    :param i:
    :return:
    """
    listOfTree = [{'no surfacing': {0: 'no', 1: {'flippers': {0: 'no', 1: 'yes'}}}},
                  {'no surfacing': {0: 'no', 1: {'flippers': {0: {'head': {0: 'no', 1: 'yes'}}, 1: 'no'}}}}]
    return listOfTree[i]


if __name__ == "__main__":
    createPlot()
