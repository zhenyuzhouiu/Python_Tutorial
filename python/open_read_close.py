path = '/home/zhenyuzhou/Desktop/darknet/data/train.txt'
#===============
# 一般使用形式
try:
    # 打开文件
    # r：读模式；w： 写模式；a：追加模式；b：二进制模式（可添加到其他模式中使用）
    # +： 读/写模式（需要添加到其他模式中使用）
    f = open(path, 'r')
    data = f.read() # 读取文件内容
# 使用finally为了确保在任何情况下，文件都能被关闭
finally:
    if f:
        f.close() # 确保文件被关闭


#===============
# 使用with语句来帮助我们自动调用close方法
with open(path, 'r') as f:
    data = f.read()
    print(data)


#===============
# readlines()和read()
# readlines() 方法会把文件每一行读入为一个字符串，并形成一个列表，在列表中每个字符串就是一行。
with open(path, 'r') as f:
    lines = f.readlines()
    line_num = len(lines)
    print(lines)
    print(line_num)

#==============
# 二进制文件，比如图片、视频等
# 在读取二进制数据时，返回的数据是字节字符串格式的,而不是文本字符串
img_path = '/home/zhenyuzhou/Desktop/darknet/data/giraffe.jpg'
with open(img_path, 'rb') as f:
    img = f.read()
    print(img)
# 读取非ASCII编码的文件，就必须以二进制模式打开，再解码。比如GBK编码文件
f = open('gbk.txt', 'rb')
u = f.read().decode('gbk')
u
print(u)
# 如果每次都这么手动转换编码嫌麻烦（写程序怕麻烦是好事，不怕麻烦就会写出又长又难懂又没法维护的代码），
# Python还提供了一个codecs模块帮我们在读文件时自动转换编码，直接读出unicode：
import codecs
with codecs.open('gbk.txt', 'r', 'gbk') as f:
    f.read()


#===============
# 逐行读取
with open(path, 'r') as f:
    while True:
        line = f.readline()
        if not line:
            break
        # 这里加了‘，’是为了避免print自动换行
        print(line),


#===============
# 文件迭代器
with open(path, 'r') as f:
    for line in f:
        print(line),