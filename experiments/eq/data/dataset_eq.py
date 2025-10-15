import torch
import random
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets, transforms

"""
Generates the dataset for xor-even task
input: (mnist image, mnist image)
output: 1 if both images not even 0 otherwise
"""

train_ds = datasets.MNIST("experiments/eq/data", train=True, download=True)

xs1 = train_ds.data[:30000].float() / 255.0
xs2 = train_ds.data[30000:].float() / 255.0

xs1 = (xs1 - 0.5) / 0.5
xs2 = (xs2 - 0.5) / 0.5

ys1 = train_ds.targets[:30000]
ys2 = train_ds.targets[30000:]

ys1 = ys1 % 2 == 0
ys2 = ys2 % 2 == 0
ys = (ys1 == ys2).long()

torch.save(xs1, "experiments/eq/data/train_eq_xs1.pt")
torch.save(xs2, "experiments/eq/data/train_eq_xs2.pt")
torch.save(ys, "experiments/eq/data/train_eq_ys.pt")

test_ds = datasets.MNIST("experiments/eq/data", train=False, download=True)

xs1 = test_ds.data[:5000].float() / 255.0
xs2 = test_ds.data[5000:].float() / 255.0

xs1 = (xs1 - 0.5) / 0.5
xs2 = (xs2 - 0.5) / 0.5

ys1 = test_ds.targets[:5000]
ys2 = test_ds.targets[5000:]

ys1 = ys1 % 2 == 0
ys2 = ys2 % 2 == 0
ys = (ys1 == ys2).long()

torch.save(xs1, "experiments/eq/data/test_eq_xs1.pt")
torch.save(xs2, "experiments/eq/data/test_eq_xs2.pt")
torch.save(ys, "experiments/eq/data/test_eq_ys.pt")
