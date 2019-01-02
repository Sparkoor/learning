import sqlite3

'''连接sqlite数据库'''
connection = sqlite3.connect("test.sqlite")
cursor = connection.cursor()
cursor.execute('''select''')
connection.commit()
connection.close()
