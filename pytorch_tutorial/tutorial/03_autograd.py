import torch

test = torch.ones(4)

if test.requires_grad:
    print("test can be captured by grad")

weights = torch.ones(4, requires_grad=True)
if weights.requires_grad:
    print("weights can be captured by grad")

optimizer = torch.optim.SGD(weights, lr=0.01)
optimizer.step()
optimizer.zero_grad()