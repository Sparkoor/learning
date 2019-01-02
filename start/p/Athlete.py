class Athlete:
    def __init__(self, name, data=None, times=[]):
        self.name = name
        self.data = data
        self.times = times

    def top3(self):
        return sorted(set(self.times))[0:3]

    def add_time(self, time=0):
        self.times.append(time)

    def add_times(self, times=[]):
        self.times.extend(times)


'''继承类'''


class NameList(list):
    def __init__(self, name, dob=None, times=[]):
        list.__init__([])
        self.name = name
        self.dob = dob
        self.extend(times)

    def top3(self):
        return sorted(set(sanitize(t) for t in self)[0:3])


def get_coach_data(filename):
    try:
        with open(filename) as f:
            data = f.readline()
            temp = data.strip().split(',')
            return NameList(temp.pop(0), temp.pop(0), temp)
    except IOError as ioerr:
        print(str(ioerr))


'''格式化数据'''


def sanitize(str):
    if '-' in str:
        spliter = '-'
    elif ':' in str:
        spliter = ':'
    else:
        return str
    (min, secs) = str.split(spliter)
    return min + '.' + secs
