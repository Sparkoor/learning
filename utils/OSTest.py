import os
# os模块的补充，带有文件复制copyfile()
import shutil

# 查看一下系统环境
print(os.environ)
print(os.environ.get('PATH'))
# 操作文件目录, 这是查询当前文件目录
absolute_path = os.path.abspath('.')
# 编写文件路径，用为不同系统的路径符不一样
new_path = os.path.join(absolute_path, 'testpath')
# 创建一个目录
# os.makedirs(new_path)
# 删除一个目录
# os.rmdir(new_path)
#  将文件名和路径分开,直接复制windows路径会报错,前缀加上r ‘r’标识‘\’不是转义字符
paths = os.path.split(r'D:\work\learning\utils\MD5.py')
print(paths[1])
# 直接得到文件拓展名
xt = os.path.splitext(r'D:\work\learning\utils\MD5.py')
print(xt[1])
# 对文件重命名
# os.rename('test.txt', 'test.py')
# 删除文件
# os.remove('test.py')
# 列出当前目录下的所有目录
y = [x for x in os.listdir('.') if os.path.isdir(x)]
print(y)
# 列出所有的文件
[x for x in os.listdir('.') if os.path.isfile(x) and os.path.splitext(x) == '.py']
