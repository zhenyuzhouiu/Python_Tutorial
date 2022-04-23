import torch

x = torch.ones(1)

w = torch.tensor(1.0, requires_grad=True)

y_hat = x * w

y = torch.tensor(2.0)

loss = (y_hat - y)**2

loss.backward()

print(w.grad)