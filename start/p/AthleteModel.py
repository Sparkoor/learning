import pickle
from Athlete import NameList, get_coach_data


# 保存数据到文件
def put_to_store(file_list):
    # 字典要先声明
    all_aths = {}
    for fi in file_list:
        ath = get_coach_data(fi)
        all_aths[ath.name] = ath
    try:
        with open("data.pickle", "wb") as store:
            pickle.dump(store, all_aths)
    except IOError as ioErr:
        print(str(ioErr))
    return all_aths


# 读取数据
def get_from_store(filename):
    aths = {}
    try:
        with open(filename) as data:
            aths = pickle.load(data)
    except IOError as ioErr:
        print(str(ioErr))
    return (aths)
