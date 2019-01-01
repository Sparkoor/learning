import sqlite3

# test.db是数据库文件，如果不存在则在当前文件夹创建
conn = sqlite3.connect('test.db')
cursor = conn.cursor()
#cursor.execute('create table user(id varchar(20) primary key,name varchar(20))')
cursor.execute('insert into user(id,name) values (?,?)', ['2', '2'])
cursor.rowcount
cursor.close()
conn.commit()
conn.close()

conn = sqlite3.connect("test.db")
cursor = conn.cursor()
cursor.execute('select * from user ')
values=cursor.fetchall()
print(values)
