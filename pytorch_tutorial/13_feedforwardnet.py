import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import transforms
import matplotlib.pyplot as plt
import torchvision
import time

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# hyper parameters
learning_rate = 0.001
num_epochs = 2
input_feature = 28*28
hidden_feature = 500
output_feature = 10
batch_size = 128

train_data = torchvision.datasets.MNIST(root="./mnist",
                                        download=True,
                                        train=True,
                                        transform=transforms.ToTensor())

test_data = torchvision.datasets.MNIST(root="./mnist",
                                       train=False,
                                       transform=transforms.ToTensor())

train_loader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True, num_workers=4)
test_loader = DataLoader(dataset=test_data, batch_size=batch_size, shuffle=True, num_workers=4)


class MnistDeepNN(nn.Module):
    def __init__(self, input_neurons, hidden_neurons, output_neurons):
        super(MnistDeepNN, self).__init__()
        self.linear1 = nn.Linear(input_neurons, hidden_neurons)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(hidden_neurons, output_neurons)

    def forward(self, x):
        x = self.relu(self.linear1(x))
        x = self.linear2(x)
        return x

model = MnistDeepNN(input_neurons=input_feature, hidden_neurons=hidden_feature, output_neurons=output_feature)
model = model.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

start_time = time.time()

n_total_steps = len(train_loader)
for epoch in range(num_epochs):
    for i, (image, label) in enumerate(train_loader):
        image = image.view(-1, input_feature)
        image = image.to(device)
        label = label.to(device)
        y_hat = model(image)
        loss = criterion(y_hat, label)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if i % 100 == 0:
            print(f'epoch: {epoch+1}/{num_epochs}, step: {i}/{n_total_steps}, loss: {loss.item():.4f}')

with torch.no_grad():
    n_sample = 0
    n_correct = 0
    for image, label in test_loader:
        image = image.view(-1, input_feature)
        image = image.to(device)
        label = label.to(device)
        y_hat = model(image)
        _, predicted = torch.max(y_hat, 1)
        n_correct += (predicted == label).sum()
        n_sample += label.shape[0]

    acc = 100* n_correct/n_sample
    print(f'Accuray of the netowork on the 10000 test images: {acc:.4f} %')

end_time = time.time()
print(f'Training and Testing time is: {end_time-start_time:.4f}s')