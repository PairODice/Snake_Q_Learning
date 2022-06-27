
def search(moves_left):
    if moves_left == 0:
        return 0
    print(moves_left)
    search(moves_left - 1)
    search(moves_left - 1)


search(2)
