'''
Transforms can be applied to PIL images, tensors, ndarrays, or custom data
during creation of the DataSet

complete list of built-in transforms:
https://pytorch.org/docs/stable/torchvision/transforms.html

On Images
---------
CenterCrop, Grayscale, Pad, RandomAffine
RandomCrop, RandomHorizontalFlip, RandomRotation
Resize, Scale

On Tensors
----------
LinearTransformation, Normalize, RandomErasing

Conversion
----------
ToPILImage: from tensor or ndrarray
ToTensor : from numpy.ndarray or PILImage

Generic
-------
Use Lambda

Custom
------
Write own class

Compose multiple Transforms
---------------------------
composed = transforms.Compose([Rescale(256),
                               RandomCrop(224)])
'''

import torch
import torchvision
from torch.utils.data import Dataset
import numpy as np

# using list to compose transforms


class WineDataset(Dataset):
    def __init__(self, transform=None):
        self.xy = np.loadtxt("wine.csv", delimiter=",", dtype=np.float32, skiprows=1)
        self.data = self.xy[:, 1:]
        self.label = self.xy[:, 0]
        self.transform = transform

    def __len__(self):
        return self.xy.shape[0]

    def __getitem__(self, item):
        trans_data, trans_label = self.data[item], self.label[item]
        if self.transform is not None:
            for t in self.transform:
                trans_data, trans_label = t(trans_data, trans_label)
            return trans_data, trans_label
        else:
            return trans_data, trans_label


class MyToTensor:
    def __call__(self, data, label):
        tensor_data = torch.from_numpy(data)
        tensor_label = torch.tensor(label)
        return tensor_data, tensor_label


class MyMulTransform:
    def __init__(self, factor):
        self.factor = factor

    def __call__(self, data, label):
        data *= self.factor
        return data, label


dataset = WineDataset()
print(dataset[0])

trans_dataset = WineDataset(transform=[MyToTensor()])
print(trans_dataset[0])

mul_dataset = WineDataset(transform=[MyMulTransform(factor=2)])
print(mul_dataset[0])

composed = [MyMulTransform(factor=2), MyToTensor()]
com_dataset = WineDataset(transform=composed)
print(com_dataset[0])

# using torchvision.transforms.Compose to compose multiple transforms


class WineDataset(Dataset):
    def __init__(self, transform=None):
        self.xy = np.loadtxt("wine.csv", delimiter=",", dtype=np.float32, skiprows=1)
        self.data = self.xy[:, 1:]
        self.label = self.xy[:, 0]
        self.transform = transform

    def __len__(self):
        return self.xy.shape[0]

    def __getitem__(self, item):
        sample = self.data[item], self.label[item]
        if self.transform is not None:
            sample = self.transform(sample)
            return sample
        else:
            return sample


class MyToTensor:
    def __call__(self, sample):
        data, label = sample
        data = torch.from_numpy(data)
        label = torch.tensor(label)
        return data, label


class MyMulTransform:
    def __init__(self, factor):
        self.factor = factor

    def __call__(self, sample):
        data, label = sample
        data *= self.factor
        return data, label


dataset = WineDataset()
print(dataset[0])

trans_dataset = WineDataset(transform=MyToTensor())
print(trans_dataset[0])

mul_dataset = WineDataset(transform=MyMulTransform(factor=2))
print(mul_dataset[0])

composed = torchvision.transforms.Compose([MyMulTransform(factor=2), MyToTensor()])
com_dataset = WineDataset(transform=composed)
print(com_dataset[0])

torch.optim.SGD()
torch