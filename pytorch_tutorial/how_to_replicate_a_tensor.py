import torch
# expand
# repeat
# torch.repeat_interleave()

data = torch.ones([2, 1, 32, 32])
test = torch.tensor([[1], [2]])
test = test.unsqueeze(-1)
test_repeat = test.repeat(2, 1, 32, 32)
test_repeate_interleave = torch.repeat_interleave(test, repeats=32, dim=1)
print(test_repeate_interleave)
test_repeate_interleave_2 = torch.repeat_interleave(test_repeate_interleave, repeats=32, dim=2)
print(test_repeate_interleave_2)

