# Using stack fullfill function of queue

class Stack():
    def __init__(self):
        self.__storage = {}
        self.__position = -1

    def push(self, value):
        self.__position += 1
        self.__storage[self.__position] = value

    def pop(self):
        if self.__position > -1:
            value = self.__storage[self.__position]
            self.__position -= 1
            return value

    def top(self):
        return self.__position