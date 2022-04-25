# What is the different between array and linked list

class Node():
    def __init__(self, value):
        self.value = value
        self.next = None

class LinkedList():
    def __init__(self):
        self.__head = None
        self.__tail = None
        self.__length = 0

    def size(self):
        return self.__length

    def add(self, value):
        node = Node(value)
        if not self.__head and not self.__tail:
            self.__head = node
            self.__tail = node
        else:
            self.__tail.next = node
            self.__tail = self.__tail.next
        self.__length += 1

    def contains(self, value):
        node = self.__head
        while node:
            if node.value == value:
                return True
            node = node.next
        return False

    def remove(self, value):
        if self.contains(value):
            current = self.__head
            previous = self.__head
            while current:
                if current.value == value:
                    if self.__head == current:
                        self.__head = self.__head.next
                        self.__length -= 1
                        return
                    if self.__tail.value == value:
                        self.__tail = previous
                    previous.next = current.next
                    self.__length -= 1

                previous = current
                current = current.next




AmazingRace = LinkedList()
AmazingRace.add("Colombo, Sri Lanka")
AmazingRace.add("Lagos, Nigeria")
AmazingRace.add("Surat, India")
AmazingRace.add("Suzhou, China")

if not AmazingRace.contains('Suzhou, China'):
    AmazingRace.add('Suzhou, China')
    print("Add the place successful!")
else:
    print("The Linked List already has the place!")

if not AmazingRace.contains('Hanoi, Vietnam'):
    AmazingRace.add('Hanoi, Vietnam')
    print("Add the place successful!")
else:
    print("The Linked List already has the place!")

print(AmazingRace._LinkedList__head.value)