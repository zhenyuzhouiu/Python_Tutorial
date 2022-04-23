import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import transforms
import matplotlib.pyplot as plt
import torchvision
import time
import sys
from torch.utils.tensorboard import SummaryWriter

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
writer = SummaryWriter(log_dir="./runs/mnist")

# hyper parameters
learning_rate = 0.001
num_epochs = 1
input_feature = 28 * 28
hidden_feature = 500
output_feature = 10
batch_size = 64

train_data = torchvision.datasets.MNIST(root="./mnist",
                                        download=True,
                                        train=True,
                                        transform=transforms.ToTensor())

test_data = torchvision.datasets.MNIST(root="./mnist",
                                       train=False,
                                       transform=transforms.ToTensor())

train_loader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True, num_workers=4)
test_loader = DataLoader(dataset=test_data, batch_size=batch_size, shuffle=False, num_workers=4)

examples = iter(test_loader)
example_data, example_target = examples.next()

image_grid = torchvision.utils.make_grid(example_data)
writer.add_image(tag="train_dataset", img_tensor=image_grid)

# for i in range(6):
#     plt.subplot(2,3,i+1)
#     plt.imshow(example_data[i][0], cmap='gray')
# plt.show()


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


model = MnistDeepNN(input_neurons=input_feature,
                    hidden_neurons=hidden_feature,
                    output_neurons=output_feature)
model = model.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

start_time = time.time()

n_total_steps = len(train_loader)
predict_accuracy = 0.0
predict_loss = 0.0
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

        predict_loss += loss.item()
        _, predict_class = torch.max(input=y_hat.data, dim=1)
        predict_accuracy += (predict_class == label).sum() / label.shape[0]



        if i % 100 == 0:
            print(f'epoch: {epoch + 1}/{num_epochs}, step: {i}/{n_total_steps}, loss: {loss.item():.4f}')
            writer.add_scalar("predict_accuracy",
                              scalar_value=predict_accuracy/100,
                              global_step= (epoch*n_total_steps+i))
            writer.add_scalar("predict_loss",
                              scalar_value=predict_loss / 100,
                              global_step=(epoch * n_total_steps + i))
            predict_accuracy = 0.0
            predict_loss = 0.0

torch.save(model.state_dict(), './checkpoints/2022-1-21-10.pth')

model.load_state_dict(torch.load('./checkpoints/2022-1-21-10.pth'))

class_labels = []
class_prob = []
with torch.no_grad():
    n_sample = 0
    n_correct = 0
    for image, label in test_loader:
        image = image.view(-1, input_feature)
        image = image.to(device)
        label = label.to(device)
        y_hat = model(image)
        _, predicted = torch.max(y_hat.data, 1)
        n_correct += (predicted == label).sum()
        n_sample += label.shape[0]

        # you should use the ground truth
        class_labels.append(label)

        # you also should use the predicted probability
        sigmoid = torch.nn.Softmax(dim=1)
        predict_prob = sigmoid(y_hat)
        class_prob.append(predict_prob)
    acc = 100 * n_correct / n_sample
    print(f'Accuray of the netowork on the 10000 test images: {acc:.4f} %')

end_time = time.time()

class_prob = torch.cat(class_prob)
class_labels = torch.cat(class_labels)
for i in range(10):
    labels_i = class_labels == i
    preds_i = class_prob[:, i]
    writer.add_pr_curve(str(i), labels_i, preds_i, global_step=0)
    writer.close()

print(f'Training and Testing time is: {end_time - start_time:.4f}s')

writer.close()