def josephus_A(n, k, m):
    people = list(range(1, n + 1))
    i = k - 1
    for num in range(n):
        count = 0
        while count < m:
            if people[i] > 0:
                count += 1
            if count == m:
                print(people[i])
                people[i] = 0
            i = (i + 1) % n
        if num < n - 1:
            print(",")
        else:
            print(" ")


##基于顺序表
def josephus_L(n, k, m):
    people = list(range(1, n + 1))

    num, i = n, k - 1
    for num in range(n, 0, -1):
        i = (i + m - 1) % num
        print(people.pop(i), end=("," if num > 1 else "\n"))
