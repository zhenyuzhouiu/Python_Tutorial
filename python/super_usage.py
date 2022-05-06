class Parent(object):
    def __init__(self):
        print("Hello, I am parent!")


class Child(Parent):
    def __init__(self):
        print("Hello, I am child!")
        # 但是父类的方法中有一部分我想使用的，还有一部分功能不存在，这就需要调用父类的重名方法。
        # 调用父类，返回一个父类对象
        # super().__init__()就是对象.方法
        super(Child, self).__init__()


son = Child()
