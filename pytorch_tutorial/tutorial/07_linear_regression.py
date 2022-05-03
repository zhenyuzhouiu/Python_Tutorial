# 1 ) Design our model
# 2 ) Construct our model loss and optimizer
# 3 ) Train our model
#    - Forward pass
#    - Backward pass
#    - Update weights

from statistics import mode
import torch
import torch.nn as nn
import numpy as np
from sklearn import datasets
import matplotlib.pyplot as plt

X_numpy, y_numpy = datasets.make_regression(n_samples=100, n_features=1, noise=20, random_state=1)

X = torch.from_numpy(X_numpy.astype(np.float32))
y = torch.from_numpy(y_numpy.astype(np.float32))
y = y.view(y.shape[0], 1)

n_sample, n_features = X.shape

input_features = n_features
output_features = 1

model = nn.Linear(input_features, output_features)
loss = nn.MSELoss()

learning_rate = 0.01
epoches = 100
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

for i in range(epoches):
    y_pred = model(X)
    l = loss(y_pred, y)
    l.backward()
    optimizer.step()
    optimizer.zero_grad()

    if epoches % 10 == 0:
        print(f"epoch:{i+1}, loss = {l.item():.4f}")


predicted = model(X).detach().numpy()
plt.plot(X_numpy, y_numpy, 'ro')
plt.plot(X_numpy, predicted, 'b')
plt.show()

