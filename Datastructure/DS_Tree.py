# Binary Tree
from DS_Queue import Queue
#===========================================================
#===Building a Binary Tree
#===========================================================
# a node should have three attribute: value, left Child, right Child
class BinaryTree():
    def __init__(self, value):
        self.value = value
        self.left_child = None
        self.right_child = None

    def insert_left(self, value):
        if not self.left_child:
            self.left_child = BinaryTree(value)
        else:
            new_node = BinaryTree(value)
            new_node.left_child = self.left_child
            self.left_child = new_node

    def insert_right(self, value):
        if not self.right_child:
            self.right_child = BinaryTree(value)
        else:
            new_node = BinaryTree(value)
            new_node.right_child = self.right_child
            self.right_child = new_node

    # DFS explores a path all the way to a leaf before backtracking and exploring another path.

    # Pre-order the middle firstly, the left second, the right last
    # recursive function
    def pre_order(self):
        print(self.value)

        if (self.left_child):
            self.left_child.pre_order()

        if (self.right_child):
            self.right_child.pre_order()

    # The left first, the middle second, and the right last.
    def in_order(self):
        if self.left_child:
            self.left_child.in_order()

        print(self.value)

        if self.right_child:
            self.right_child.in_order()

    # The left first, the right second, and the middle last.
    def post_order(self):
        if self.left_child:
            self.left_child.post_order()

        if self.right_child:
            self.right_child.pre_order()

        print(self.value)

    # BFS algorithm traverses the tree level by level and depth by depth.
    def breadth_fs(self):
        b_queue = Queue()
        b_queue.enqueue(self)

        while b_queue.size():
            curr_node = b_queue.dequeue()
            print(curr_node.value)

            if curr_node.left_child:
                b_queue.enqueue(curr_node.left_child)

            if curr_node.right_child:
                b_queue.enqueue(curr_node.right_child)

class Binary_Search_Tree():
    def __init__(self):
        self.value
        self.left_child = None
        self.right_child = None

    def insert(self, value):
        return 0
            


a_node = BinaryTree(1)
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

# print(a_node.value) # a
# print(b_node.value) # b
# print(c_node.value) # c
# print(d_node.value) # d
# print(e_node.value) # e
# print(f_node.value) # f
# print(g_node.value) # g


a_node.pre_order()
print("==================================================")
a_node.in_order()
print("==================================================")
a_node.post_order()
print("==================================================")
a_node.breadth_fs()


