import torch
import random
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets, transforms
from torchvision.transforms.functional import rotate
import kornia

"""
Generates the dataset for conditional '1' task
"""

def dilate_n_td(x, n):
    filt = torch.tensor([
    [0.,1,0.],
    [0.,1,0.],
    [0.,1,0.]])
    for i in range(n):
        x_dilated = kornia.morphology.dilation(x.view(1,1,28,28), filt)
        # after dilating the image we need to
        # interpolate
        x = torch.lerp(x, x_dilated, .7)
    return x

def dilate_n_lr(x, n):
    filt = torch.tensor([
    [0.,0,0.],
    [1.,1,1.],
    [0.,0,0.]])
    for i in range(n):
        x_dilated = kornia.morphology.dilation(x.view(1,1,28,28), filt)
        # after dilating the image we need to
        # interpolate
        x = torch.lerp(x, x_dilated, .7)
    return x

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

train_ds = datasets.MNIST("experiments/dilate/data", train=True, download=True)
test_ds = datasets.MNIST("experiments/dilate/data", train=False, download=True)

xs_train = []
ys_train = []
for x, label in train_ds:
    x = transform(x)
    dilated = dilate_n_td(x, label)
    y = dilated

    xs_train.append(x)
    ys_train.append(y.view(1,28,28))

xs_train = torch.cat(xs_train)
ys_train = torch.cat(ys_train)

xs_test = []
ys_test = []
for x, label in test_ds:
    x = transform(x)
    dilated = dilate_n_td(x, label)
    y = dilated

    xs_test.append(x)
    ys_test.append(y.view(1,28,28))

xs_test = torch.cat(xs_test)
ys_test = torch.cat(ys_test)

torch.save(xs_train, "experiments/dilate/data/train_xs.pt")
torch.save(ys_train, "experiments/dilate/data/train_ys.pt")
torch.save(xs_test, "experiments/dilate/data/test_xs.pt")
torch.save(ys_test, "experiments/dilate/data/test_ys.pt")

# Template for generating example figures
# ----------------------------------------
# plt.figure(figsize=(4,4), dpi=300)
# plt.imshow(xs_test[10],cmap="grey")
# plt.axis("off")
# plt.savefig("experiments/dilate/figures/x1_test.pdf", format="pdf", bbox_inches="tight", pad_inches=0)
# plt.show()



