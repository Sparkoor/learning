"""
  ORM框架
"""


class user2(object):
    def __init__(self, id, name):
        self.id = id
        self.name = name


# 导入sqlalchemy，并初始化DBSession
from sqlalchemy import Column, String, create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.declarative import declarative_base

# 创建对象基类
Base = declarative_base()


# 定义user对象
class user2(Base):
    # 表的名称
    __tablename__ = 'user2'
    # 表的结构
    id = Column(String(20), primary_key=True)
    name = Column(String(20))


# 初始化数据库连接 '数据库类型+数据库驱动名称://用户名:口令@机器地址:端口号/数据库名'
engine = create_engine('mysql+mysqlconnector://root:1122@localhost:3306/programs')
# 创建DBSession
DBSession = sessionmaker(bind=engine)

# 创建session对象
session = DBSession()
# 创建新的user对象
new_user = user2()
new_user.id = '4'
new_user.name = 'tom'
# 添加
session.add(new_user)
# 提交保存
session.commit()
session.close()

# 查询 .one() .all()
session = DBSession()
user = session.query(user2).filter(user2.id == '1').one()

print(type(user))
print(user.name)
