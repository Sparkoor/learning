import pandas as pd
import numpy as np

file_read = pd.read_csv('dataset2.csv', delimiter='\s+')
print(file_read)
print(type(file_read))
print(file_read.shape)
a = file_read.head(3)
b = file_read.tail(4)
print(a)
print('sss')
print(b)
c = file_read.loc[2:4]
d = file_read['num'].max()
print('d')
print(d)
col = file_read.columns.tolist()
print(col)
# 排序 默认升序
file_read.sort_values("num", inplace=True, ascending=False)
# 用来过滤数据
colum_is_null = file_read['num'].isnull()
print(colum_is_null)
# 过滤
mean_nm = file_read['num'][colum_is_null == False]
print(mean_nm)
# 求均值
print(mean_nm.mean())
# 分组求值    以色泽为组， num是被计算的值可以是个数组， aggfunc是调用的函数
group = file_read.pivot_table(index="色泽", values="num", aggfunc=np.mean)
# 填充值 横向替换 axis=1纵向替换
file_read['num'].fillna(axis=0, method='ffill')
# 删除缺失值
new_data = file_read.dropna(axis=0, subset=['num', '色泽'])
print(new_data)
# 直接定位
dta = file_read.loc[2, "num"]
# 根据num排序
new_sort = file_read.sort_values('num', ascending=True)
# 获取列名
print(list(file_read))
print(list(file_read.columns.values))
print(list(file_read.columns.tolist()))
# 重新设置index值 drop 以前的索引值删除
new_sort.reset_index(drop=True)


# 自定义函数，类似于map的形式new_fuc,loc[5]是定位到索引值是5的位置
def five_row(column):
    five_item = column.loc[5]
    return five_item


# 传入函数，然后每一列作用在
new_fuc = file_read.apply(five_row)
print(file_read)
print(new_fuc)
# 读取到的数据是dataFrame结构
print(type(file_read))
row = file_read['num']
# 这一列是Series
print(type(row))
# 这是ndarray
print(type(row.values))
# 构造一个Series结构
