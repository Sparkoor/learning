aa = 1
if True:
    print("aaa")
else:
    print("bbb")

movies = ['monkey', 1922, ['json', 'tim', ['haizi', 'wulei', 'meili']]]


def print_lol(movies_list):
    for msg in movies_list:
        if isinstance(msg, list):
            print_lol(msg)
        else:
            print(msg)


print_lol(movies)

data = open('data.txt')
if 'data' in locals():
    print(str(locals()))
else:
    print("no data")
