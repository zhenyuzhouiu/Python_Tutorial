# convolution is round down
# pooling is round up


import torch
import torchvision
from torch.utils.data import DataLoader

train_data = torchvision.datasets.CIFAR10(root='./CIFAR10', download=True, train=True, transform=torchvision.transforms.ToTensor())

train_loader = DataLoader(dataset=train_data, batch_size=4, shuffle=True, num_workers=4)

X = iter(train_loader)
image, label = next(X)
print(image[0].shape)
conv = torch.nn.Conv2d(in_channels=3, out_channels=6, stride=2, kernel_size=5, padding=0)
test = image[0].view(1, 3, 32, 32)
output = conv(test)
print(output.shape)

pool = torch.nn.MaxPool2d(kernel_size=3, stride=2)
output = pool(test)
print(output.shape)