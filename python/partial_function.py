from functools import partial


def add(*args):
    return sum(args)


add_100 = partial(add, 100)
print(add_100(1, 2, 3))
