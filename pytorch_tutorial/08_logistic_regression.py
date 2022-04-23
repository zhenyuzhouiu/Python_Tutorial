# 1 ) Design our model
# 2 ) Construct our model loss and optimizer
# 3 ) Train our model
#    - Forward pass
#    - Backward pass
#    - Update weights

import torch
import torch.nn as nn
import numpy as np
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Data Processing
breast_cancer = datasets.load_breast_cancer()

# the data type is double, so we should transform it to float32
X_train, X_test, y_train, y_test = train_test_split(breast_cancer.data, breast_cancer.target, test_size=0.2,
                                                    random_state=118)
# Scalar
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


X_train = torch.from_numpy(X_train.astype(np.float32))
y_train = torch.from_numpy(y_train.astype(np.float32))
y_train = y_train.view(y_train.shape[0], 1)

n_sample, n_features = X_train.shape

X_test = torch.from_numpy(X_test.astype(np.float32))
y_test = torch.from_numpy(y_test.astype(np.float32))
y_test = y_test.view(y_test.shape[0], 1)


class SimpleLogisticRegression(nn.Module):
    def __init__(self, input_features, output_features):
        super(SimpleLogisticRegression, self).__init__()
        self.linear = nn.Linear(input_features, output_features)

    def forward(self, x):
        y_predicted = torch.sigmoid(self.linear(x))
        return y_predicted


learning_rate = 0.01
model = SimpleLogisticRegression(n_features, 1)
loss = nn.BCELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
epochs = 100

for epoch in range(epochs):
    y_hat = model(X_train)
    l = loss(y_hat, y_train)
    l.backward()
    optimizer.step()
    optimizer.zero_grad()
    if epoch % 10 == 0:
        print(f'epoche: {epoch + 1}, loss: {l.item():.4f}')


with torch.no_grad():
    y_hat = model(X_test)
    cancer = y_hat.round()
    accuracy = cancer.eq(y_test).sum() / float(y_test.shape[0])
    print(f'after epoches: {epochs}, the accuracy on testing dataset: {accuracy}')
