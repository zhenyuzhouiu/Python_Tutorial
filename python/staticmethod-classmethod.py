# 正确理解Python中的 @staticmethod@classmethod方法
# https://zhuanlan.zhihu.com/p/28010894

class A(object):
    def m1(self, n):
        print("self:", self)

    @classmethod
    def m2(cls, n):
        print("cls:", cls)

    @staticmethod
    def m3(n):
        pass

a = A()
a.m1(1) # self: <__main__.A object at 0x000001E596E41A90>
A.m2(1) # cls: <class '__main__.A'>
A.m3(1)
# A.m1 <unbound method A.m1>
print(A.m1)
print(a.m1)
# A.m1(a, 1) 等价 a.m1(1)
A.m1(1)