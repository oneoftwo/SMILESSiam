import torch 
from torch import nn

a = torch.rand(1, 32)
b = torch.rand(1, 32)

ab = torch.cat([a, b], dim=1)
print(ab.size())

n = nn.Sequential(
        nn.Linear(64, 64),
        nn.ReLU(),
        # nn.BatchNorm1d(64),
        nn.Linear(64, 64)
        )

n(ab)

