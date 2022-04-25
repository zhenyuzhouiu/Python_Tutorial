
class BFSTree:
    def __init__(self, value):
        self.value = value
        self.left_child = None
        self.right_child = None

    def insert_left(self, value):
        if self.left_child is None:
            self.left_child = BFSTree(value)
        else:
            new_node = BFSTree(value)
            new_node.left_child = self.left_child
            self.left_child = new_node

    def insert_right(self, value):
        if self.right_child is None:
            self.right_child = BFSTree(value)
        else:
            new_node = BFSTree(value)
            new_node.right_child = self.right_child
            self.right_child = new_node

    def bfs(self):
        queue = BFSQueue()
        queue.enqueue(self)

        while queue.queue_len():
            current_node = queue.dequeue()
            print(current_node.value)

            if current_node.left_child is not None:
                queue.enqueue(current_node.left_child)

            if current_node.right_child is not None:
                queue.enqueue(current_node.right_child)


class BFSQueue:
    def __init__(self):
        self.start = -1
        self.end = -1
        self.storage = {}

    def queue_len(self):
        return abs(self.end - self.start)

    def enqueue(self, value):
        self.end += 1
        self.storage[self.end] = value

    def dequeue(self):
        if self.end > self.start:
            self.start += 1
            value = self.storage[self.start]
            self.storage.pop(self.start)
            return value


a_node = BFSTree(1)
a_node.insert_left(2)
a_node.insert_right(5)

b_node = a_node.left_child
b_node.insert_left(3)
b_node.insert_right(4)

d_node = b_node.left_child
e_node = b_node.right_child

c_node = a_node.right_child
c_node.insert_left(6)
c_node.insert_right(7)

f_node = c_node.left_child
g_node = c_node.right_child
#
# print(a_node.value) # a
# print(b_node.value) # b
# print(c_node.value) # c
# print(d_node.value) # d
# print(e_node.value) # e
# print(f_node.value) # f
# print(g_node.value) # g

a_node.bfs()

