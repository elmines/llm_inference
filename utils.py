
def list_rfind(l, x):
    for i in range(len(l) - 1, -1, -1):
        if l[i] == x:
            return i
    return -1