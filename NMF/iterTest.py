"""
迭代器的使用
"""
import os, time


class loadFolders(object):
    def __init__(self, path):
        self.path = path

    def __iter__(self):
        for file in os.listdir(self.path):
            file_abspath = os.path.join(self.path, file)
            if os.path.isdir(file_abspath):
                yield file_abspath


class loadFiles(object):
    def __init__(self, par_path):
        self.par_path = par_path

    def __iter__(self):
        folders = loadFolders(self.par_path)
        for folder in folders:
            # os.sep 文件分割符
            catg = folder.split(os.sep)[-1]
            for file in os.listdir(folder):
                yield catg, file


if __name__ == '__main__':
    filepath = os.path.abspath(r'../data/')
    files = loadFiles(filepath)
    # 在这用完一遍，循环一遍
    for i, msg in enumerate(files):
        print(i)
