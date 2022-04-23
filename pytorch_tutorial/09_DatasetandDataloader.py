import torch
import numpy as np
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import math

class WineDataset(Dataset):
    def __init__(self):
        self.xy = np.loadtxt("wine.csv", delimiter=",", skiprows=1)
        self.data = torch.from_numpy(self.xy[:, 1:].astype(np.float32))
        self.label = torch.from_numpy(self.xy[:, 0].astype(np.float32))

    def __len__(self):
        return self.xy.shape[0]

    def __getitem__(self, item):
        return self.data[item], self.label[item]


wine_dataset = WineDataset()
wine_dataloader = DataLoader(wine_dataset, batch_size=4, shuffle=True, num_workers=4)
wine_data, wine_label = next(iter(wine_dataloader))

n_sample, n_feature = wine_dataset.xy.shape
mini_sample, mini_feature = wine_data.shape
num_iter = math.ceil(n_sample/mini_sample)
print(n_sample, num_iter)

num_epochs = 2

for epoch in range(num_epochs):
    for i, (wine_data, wine_label) in enumerate(wine_dataloader):
        if (i+1)%5==0:
            print(f'epoch {epoch+1}/{num_epochs}, step {i+1}/{num_iter}, input {wine_data.shape}')
