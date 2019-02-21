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
    """
    计算数据类型
    :param myTree:
    :param parentPt:
    :param nodeTxt:
    :return:
    """
    numLeafs = getNumLeafs(myTree)
    numDepth = getDept(myTree)
    firstStr = list(myTree.keys())[0]
    # plotTree.xOff plotTree.yOff 用来追踪已绘制节点的位置 y是固定的表示，同一行的孩子
    cntrPt = (plotTree.xOff + (1.0 + float(numLeafs)) / 2.0 / plotTree.totalW, plotTree.yOff)
    plotMidText(cntrPt, parentPt, nodeTxt)
    plotNode(firstStr, cntrPt, parentPt, decisinNode)
    secondDict = myTree[firstStr]
    plotTree.yOff = plotTree.yOff - 1.0 / plotTree.totalD
    for key in secondDict.keys():
        if type(secondDict[key]).__name__ == 'dict':
            plotTree(secondDict[key], cntrPt, str(key))
        else:
            plotTree.xOff = plotTree.xOff + 1.0 / plotTree.totalW
            plotNode(secondDict[key], (plotTree.xOff, plotTree.yOff), cntrPt, leafNode)
            plotMidText((plotTree.xOff, plotTree.yOff), cntrPt, str(key))
    # 返回上一个坐标
    plotTree.yOff = plotTree.yOff + 1.0 / plotTree.totalD


def createPlot(inTree):
    """
    创建画布
    :return:
    """
    fig = plt.figure(1, facecolor='white')
    fig.clf()
    # 某个设置
    axprops = dict(xticks=[], yticks=[])
    # 这是一种全局变量
    createPlot.ax1 = plt.subplot(111, frameon=False, **axprops)
    plotTree.totalW = float(getNumLeafs(inTree))
    plotTree.totalD = float(getDept(inTree))
    plotTree.xOff = -0.5 / plotTree.totalW
    plotTree.yOff = 1.0
    plotTree(inTree, (0.5, 1.0), "")
    plt.show()


def getNumLeafs(myTree):
    """
    获取树的叶的数量
    :param myTree:
    :return:
    """
    numLeafs = 0
    firstStr = list(myTree.keys())[0]
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
    # 获取key值需要转化成list
    firstChild = list(myTree.keys())[0]
    secondDict = myTree[firstChild]
    for key in secondDict.keys():
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
    createPlot(retrieveTree(0))
