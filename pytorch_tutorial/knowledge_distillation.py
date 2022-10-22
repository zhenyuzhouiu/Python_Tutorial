import torch
from torch import nn
import torch.nn.functional as F
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader
from torchinfo import summary
from tqdm import tqdm

# 设置随机数种子，便于复现
torch.manual_seed(0)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 使用cuDNN加速卷积运算
torch.backends.cudnn.benchmark = True

train_dataset = torchvision.datasets.MNIST(
    root="./pytorch_tutorial/dataset/",
    train=True,
    transform=transforms.ToTensor(),
    download=True
)

test_dataset = torchvision.datasets.MNIST(
    root="./pytorch_tutorial/dataset/",
    train=False,
    transform=transforms.ToTensor(),
    download=True
)

train_loader = DataLoader(dataset=train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=32, shuffle=False)

# 教师模型
class TeacherModel(nn.Module):
    def __init__(self, in_channels=1, num_classes=10):
        super(TeacherModel, self).__init__()
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(784, 1200)
        self.fc2 = nn.Linear(1200, 1200)
        self.fc3 = nn.Linear(1200, num_classes)
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, x):
        x = x.view(-1, 784)

        x = self.fc1(x)
        x = self.dropout(x)
        x = self.relu(x)

        x = self.fc2(x)
        x = self.dropout(x)
        x = self.relu(x)

        x = self.fc3(x)

        return x


model = TeacherModel()
model = model.to(device)
summary(model)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

epoches = 6
for epoch in range(epoches):
    model.train()

    for data, targets in tqdm(train_loader):
        data = data.to(device)
        targets = targets.to(device)

        preds = model(data)
        loss = criterion(preds, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    model.eval()
    num_correct = 0
    num_samples = 0

    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(device)
            y = y.to(device)

            preds = model(x)
            predictions = preds.max(1).indices
            num_correct += (predictions == y).sum()
            num_samples += predictions.size(0)
        acc = (num_correct/num_samples).item()

    model.train()
    print('Epoch:{} \t Accuracy"{:.4f}'.format(epoch+1, acc))

teacher_model = model


class StudentModel(nn.Module):
    def __init__(self, in_channels=1, num_classes=10):
        super(StudentModel, self).__init__()
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(784, 20)
        self.fc2 = nn.Linear(20, 20)
        self.fc3 = nn.Linear(20, num_classes)

    def forward(self, x):
        x = x.view(-1, 784)
        x = self.fc1(x)
        x = self.relu(x)

        x = self.fc2(x)
        x = self.relu(x)

        x = self.fc3(x)

        return  x


model = StudentModel()
model = model.to(device)
summary(model)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

epoches = 6
for epoch in range(epoches):
    model.train()

    for data, targets in tqdm(train_loader):
        data = data.to(device)
        targets = targets.to(device)

        preds = model(data)
        loss = criterion(preds, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    model.eval()
    num_correct = 0
    num_samples = 0

    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(device)
            y = y.to(device)

            preds = model(x)
            predictions = preds.max(1).indices
            num_correct += (predictions == y).sum()
            num_samples += predictions.size(0)
        acc = (num_correct/num_samples).item()

    model.train()
    print('Epoch:{} \t Accuracy"{:.4f}'.format(epoch+1, acc))

student_model_scratch = model

teacher_model.eval()
model = StudentModel()
model = model.to(device)
model.train()
temp = 7

hard_loss = nn.CrossEntropyLoss()
alpha = 0.2

soft_loss = nn.KLDivLoss(reduction='batchmean')
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

epoches = 6
for epoch in range(epoches):
    model.train()

    for data, targets in tqdm(train_loader):
        data = data.to(device)
        targets = targets.to(device)

        with torch.no_grad():
            teacher_preds = teacher_model(data)

        student_preds = model(data)
        student_loss = hard_loss(student_preds, targets)
        distilation_loss = soft_loss(F.softmax(student_preds/temp, dim=1),
                                     F.softmax(teacher_preds/temp, dim=1))

        loss = alpha * student_loss + (1-alpha) * distilation_loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    model.eval()
    num_correct = 0
    num_samples = 0

    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(device)
            y = y.to(device)

            preds = model(x)
            predictions = preds.max(1).indices
            num_correct += (predictions == y).sum()
            num_samples += predictions.size(0)
        acc = (num_correct/num_samples).item()

    model.train()
    print('Epoch:{} \t Accuracy"{:.4f}'.format(epoch+1, acc))