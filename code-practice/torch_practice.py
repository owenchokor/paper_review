import torch
import torch.nn as nn

a = torch.randn((3, 5))
b = torch.LongTensor([0, 1, 4])

loss_fn = nn.CrossEntropyLoss()
print(loss_fn(a, b))