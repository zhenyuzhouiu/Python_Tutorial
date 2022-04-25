# An important property of a binary search tree is that the value of a binary
# search tree node is larger than the value of the offspring of its left child,
# but smaller than the value of the offspring of its right child.

class BinarySearchTree:
    def __init__(self, value):
        self.value = value
        self.left_child = None
        self.right_child = None

    def insert_node(self, value):
        if self.value >= value:
            if self.left_child is None:
                self.left_child = BinarySearchTree(value)
            else:
                self.left_child.insert_node(value)

        if self.value < value:
            if self.right_child is None:
                self.right_child = BinarySearchTree(value)
            else:
                self.right_child.insert_node(value)

    def find_node(self, value):
        if self.value < value and self.right_child:
            return self.right_child.find_node(value)
        if self.value > value and self.left_child:
            return self.left_child.find_node(value)

        return value == self.value

    def remove_node(self, value):
        


bst = BinarySearchTree(15)
bst.insert_node(10)
bst.insert_node(8)
bst.insert_node(12)
bst.insert_node(20)
bst.insert_node(17)
bst.insert_node(25)
bst.insert_node(19)
print(bst.find_node(0)) # False
print(bst.find_node(19))