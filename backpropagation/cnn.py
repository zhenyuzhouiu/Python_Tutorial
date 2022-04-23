# This code aims to show that how to update weights of cnn kernel

import torch
import numpy as np

cnn = torch.nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3, stride=1, padding=0)
torch.nn.init.constant_(cnn.weight, 0.)
torch.nn.init.constant_(cnn.bias, 0.)

print("CNN Weight Initial: ", cnn.weight.data)
optim = torch.optim.SGD(cnn.parameters(), lr=0.1)

x = np.array([[2.0, 2.0, 2.0, 2.0], [2.0, 2.0, 2.0, 2.0], [2.0, 2.0, 2.0, 2.0]])
x = torch.from_numpy(x)
x= x.unsqueeze(dim=0).unsqueeze(dim=0)
x.requires_grad = True
x = x.double()

cnn = cnn.double()


label = np.ones([1, 2])
label = torch.from_numpy(label)
label = label.unsqueeze(dim=0).unsqueeze(dim=0)
label = label.double()
for epoch in range(10):

    y = cnn(x)
    mse_loss = torch.sum((y - label)**2)

    mse_loss.backward()
    optim.step()
    print("CNN Weight Trained: ", cnn.weight.data)
    optim.zero_grad()



