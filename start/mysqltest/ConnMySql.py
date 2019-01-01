"""
pip install mysql-connector-python --allow-external mysql-connector-python
pip install mysql-connector
"""
# 导入mysql驱动
import mysql.connector

conn = mysql.connector.connect(user='root', password='1122', database='programs')
# cursor = conn.cursor()
# # 创建user表
# cursor.execute('create table user2(id varchar(20),name varchar(20))')
# cursor.execute('insert into user2(id,name) value(%s,%s)', ['1', '1'])
# cursor.rowcount
# # 提交事务
# conn.commit()
# cursor.close()
# 查询
cursor = conn.cursor()
cursor.execute("select * from user2 ")
values = cursor.fetchall()
print(values)
cursor.close()
conn.close()
