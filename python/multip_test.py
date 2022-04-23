# multiprocessing包下的Queue是多进程安全的队列，我们可以通过该Queue来进行多进程之间的数据传递。

from multiprocessing import Pool, Queue


import random
import time
import multiprocessing


def worker(name, q):
    t = 0
    for i in range(10):
        print(name + " " + str(i))
        x = random.randint(1, 3)
        t += x
        time.sleep(x * 0.1)
    q.put(t)

if __name__ == "__main__":
    q = Queue()
    jobs = []
    for i in range(10):
        p = multiprocessing.Process(target=worker, args=(str(i), q))
        jobs.append(p)
        p.start()

    for p in jobs:
        p.join()

    results = [q.get() for j in jobs]
    print(results)

