# using torch
# 1 ) Design our model
# 2 ) Construct our model loss and optimizer
# 3 ) Train our model
#    - Forward pass
#    - Backward pass
#    - Update weights

import torch
import torch.nn as nn

x = torch.tensor([[1], [2], [3], [4]], dtype=torch.float32)
y = torch.tensor([[2], [4], [6], [8]], dtype=torch.float32)

num_sample, in_feature = x.shape
num_sample, out_feature = y.shape


class SimpleLinearRegression(nn.Module):
    def __init__(self, input_features, output_features):
        super(SimpleLinearRegression, self).__init__()
        self.linear = nn.Linear(input_features, output_features)

    def forward(self, x):
        return self.linear(x)


model = SimpleLinearRegression(in_feature, out_feature)
loss = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

epoch = 100

x_test = torch.tensor([5], dtype=torch.float32)
print(f'Prediction before training: f(5) = {model(x_test).item():.3f}')

for i in range(epoch):
    # forward pass
    y_hat = model(x)
    # loss
    l = loss(y_hat, y)
    # backward pass
    l.backward()
    # update weights
    optimizer.step()
    # zero gradients
    optimizer.zero_grad()

    if epoch % 10 == 0:
        [w, b] = model.parameters()
        print(f'epoch {i+1}: w = {w[0][0].item():.3f}, loss={l:.8f}')

print(f'Prediction after training: f(5) = {model(x_test).item():.3f}')

