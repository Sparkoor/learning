import pandas as pd
import os
import networkx as nx
import matplotlib.pyplot as plt

os.chdir('dataset')

data1 = pd.read_excel('Agraph.xlsx')
data2 = pd.read_excel('Bgraph.xlsx')
data3 = pd.read_excel('Cgraph.xlsx')
data4 = pd.read_excel('Dgraph.xlsx')
data = pd.concat([data1, data2, data3, data4])

print(data)
# 仅打印前几个
print(data.head())
# networkx封装好的图对象
graph = nx.from_pandas_edgelist(data, 'Source', 'Target', edge_attr=['Weight'])

print(graph)
"""度中心：一个节点在网络中的连接数"""
degree_df = pd.DataFrame(columns=['dcenter'])
degree_df['dcenter'] = pd.Series(nx.degree_centrality(graph))
degree_df.sort_values('dcenter', ascending=False)[:10].plot(kind='barh')
plt.show()
"""接近中心性：具有高接近中心性的节点通常在集群之间被高度连接"""
"""中介中心性：识别连接不同的节点"""
"""网页排名：找出活跃用户"""
"""
社交网络的可视化
"""
plt.figure(figsize=(20, 10))
# 节点的颜色由节点的度决定，
node_color = [graph.degree(v) for v in graph]
# 节点的大小由度的中心性决定
node_size = [5000 * nx.degree_centrality(graph)[v] for v in graph]
# 边的宽度由权重决定
edge_width = [0.2 * graph[u][v]['Weight'] for u, v in graph.edges()]
# 以某种方式表示
pos = nx.spring_layout(graph)
# 画图
nx.draw_networkx(graph, pos, node_size=node_size, node_color=node_color, alpha=0.7, with_labels=False,
                 width=edge_width)
# 最重要的top10人物
top = degree_df.sort_values('dcenter', ascending=False)[:10]
top = top.index.values.tolist()
# 创建label，不同的度量方法，产生不同的标签
labels = {role: role for role in top}
# 添加label
nx.draw_networkx_labels(graph, pos, labels=labels, font_size=20)
plt.show()
nx.write_gexf(graph, 'dcenter.gexf')

"""最小社区探测"""
k = 3
klist = [['A','C'], 'B']
print(klist)
plt.figure(figsize=(20, 10))
nx.draw_networkx(graph, pos, nodelist=klist[0], node_size=node_size, node_color='r', alpha=0.7, with_labels=False,
                 width=edge_width)
nx.draw_networkx(graph, pos, nodelist=klist[1], node_size=node_size, node_color='y', alpha=0.7, with_labels=False,
                 width=edge_width)
nx.draw_networkx_labels(graph, pos, labels=labels, font_size=20)
plt.show()
