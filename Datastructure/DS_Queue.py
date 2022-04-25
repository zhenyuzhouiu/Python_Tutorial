# FIFO first input first output

class Queue:
    def __init__(self):
        self.__storage = {}
        self.__start = -1 #replicating 0 index used for arrays
        self.__end = -1 #replicating 0 index used for arrays

    def enqueue(self, value):
        self.__end += 1
        self.__storage[self.__end] = value

    def dequeue(self):
        if self.size():
            self.__start += 1
            value = self.__storage[self.__start]
            del self.__storage[self.__start]
            # we need reset the number of start and end
            if not self.size():
                self.__end = -1
                self.__start = -1

            return value

    def size(self):
        return self.__end - self.__start

microsoftQueue = Queue()
microsoftQueue.enqueue("{user: ILoveWindows@gmail.com}")
microsoftQueue.enqueue("{user: cortanaIsMyBestFriend@hotmail.com}")
microsoftQueue.enqueue("{user: InternetExplorer8Fan@outlook.com}")
microsoftQueue.enqueue("{user: IThrowApplesOutMyWindow@yahoo.com}")

print(microsoftQueue._Queue__storage)