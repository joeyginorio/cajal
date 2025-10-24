import torch
import random
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets, transforms
from torchvision.transforms.functional import gaussian_blur


"""
Generates the dataset for conditional '1' task
"""

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

train_ds = datasets.MNIST("experiments/blur/data", train=True, download=True)
test_ds = datasets.MNIST("experiments/blur/data", train=False, download=True)

xs_train = []
ys_train = []
for x, label in train_ds:
    x = transform(x)
    blurred = x
    for i in range(0,label):
        blurred = gaussian_blur(blurred, (13,13), sigma=(1,1))

    xs_train.append(x)
    ys_train.append(blurred)

xs_train = torch.cat(xs_train)
ys_train = torch.cat(ys_train)

xs_test = []
ys_test = []
for x, label in test_ds:
    x = transform(x)
    blurred = x
    for i in range(0,label):
        blurred = gaussian_blur(blurred, (13,13), sigma=(1,1))

    xs_test.append(x)
    ys_test.append(blurred)

xs_test = torch.cat(xs_test)
ys_test = torch.cat(ys_test)

torch.save(xs_train, "experiments/blur/data/train_xs.pt")
torch.save(ys_train, "experiments/blur/data/train_ys.pt")
torch.save(xs_test, "experiments/blur/data/test_xs.pt")
torch.save(ys_test, "experiments/blur/data/test_ys.pt")



