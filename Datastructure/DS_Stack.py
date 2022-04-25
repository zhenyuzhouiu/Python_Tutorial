# LIFO last input last output

class Stack():
    def __init__(self):
        self.__storage = {}
        self.__position = -1

    def push(self, value):
        self.__position += 1
        self.__storage[self.__position] = value

    def pop(self):
        if self.__position > -1:
            val = self.__storage[self.__position]
            del self.__storage[self.__position]
            self.__position -= -1
            return val

    def top(self):
        return self.__position


browserHistory = Stack()
forward = Stack()
browserHistory.push("google.com")
browserHistory.push("medium.com")
browserHistory.push("freecodecamp.com")
browserHistory.push("netflix.com")
print(browserHistory._Stack__storage)

val = browserHistory.pop()
forward.push(val)
print(val)
print(browserHistory._Stack__storage)
print(forward._Stack__storage)