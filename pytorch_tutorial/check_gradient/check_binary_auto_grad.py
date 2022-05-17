import torch
import matplotlib.pyplot as plt
import matplotlib.image as mpimage
from torch.autograd import Variable

for i in range(100):
    image = mpimage.imread("/home/zhenyuzhou/Desktop/Finger-Knuckle-Recognition/"
                           "dataset/PolyUKnuckleV3/Session_1_128/1-104/1/1.bmp")
    image = torch.from_numpy(image[:, :, 0]) / 255.0
    threshold = torch.rand([1], requires_grad=True)
    # if you use torch.where() it will return a tuple instead of tensor
    sign_binary = torch.sign(image - threshold)
    binary = torch.relu(sign_binary)

    f, axarr = plt.subplots(1, 2)
    axarr[0].imshow(image, "gray")
    axarr[0].set_title("Dataset Images")
    axarr[1].imshow(binary.detach().cpu(), "gray")
    axarr[1].set_title("Transformer Images")

    plt.ioff()
    plt.show()

