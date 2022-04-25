# Using an Array to represent a Circular Queue

class Circular_Queue():
    def __init__(self, maxsize):
        self.__storage = {}
        self.__start = -1
        self.__end = -1
        self.maxsize = maxsize

    def isFull(self):
        if ((self.__end + 1) % self.maxsize) == (self.__start % self.maxsize):
            return True
        else:
            return False

    def isEmpty(self):
        if self.__end == self.__start:
            return True
        else:
            return False

    def addToQ(self, value):
        if not self.isFull():
            self.__end = (self.__end + 1) % self.maxsize
            self.__storage[self.__end] = value
        else:
            print("The Circular Queue is Full")

    def deleteFromQ(self):
        if not self.isEmpty():
            value = self.__storage[self.__start]
            del self.__storage[self.__start]
            self.__start = (self.__start + 1) % self.maxsize
        else:
            print("The Circular Queue is Empty")

    def clearQ(self):
        self.__storage.clear()
        self.__start = -1
        self.__end = -1
