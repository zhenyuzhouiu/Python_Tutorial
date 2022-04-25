#==============
# with * as *
# python context manager protocol: __enter__ and __exit__
#==============
class Sample:
    # when "with" is executed, it will trigger this method
    def __enter__(self):
        print("in __enter__")
        return "Foo"

    # when "with" ends, it will trigger the operation
    # exc_type: the exception type if wrong
    # exc_val: the exception content
    # exc_tb: the position of exception
    def __exit__(self, exc_type, exc_val, exc_tb):
        print("in __exit__")

def get_sample():
    return Sample()

with get_sample() as sample:
    print("Sample: ", sample)


#===============
# zip()
# is used to take an iterable object as a parameter, pack the corresponding elements
# in the object into tuples, and then return a list of these tuples
#===============
a = [1, 2, 3]
b = [4, 5, 6]
c = [4, 5, 6, 7, 8]
zipped1 = zip(a, b)
print(list(zipped1))
zipped2 = zip(a, c)
print(list(zipped2))
re_zipped = zip(*zipped1)
print(list(re_zipped))


#===============
# 使用multiprocessing来实现多进程
#===============
from multiprocessing import Process

def f(name):
    print('hello', name)
p = Process(target=f, args=('xiaoshuaib', ))
p.start()
p.join()
# 还可以使用进程池的方式
from multiprocessing import Pool

def f(x):
    return x*x
with Pool(5) as p:
    print(p.map(f, [1,2,3]))