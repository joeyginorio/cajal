import os
import torch
import random
import csv
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import cajal.syntax as cj
from cajal.compiling import compile, TypedTensor
import cajal.typing as cj
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets, transforms
# from mnist_cnn.mnist import CNN
from torcheval.metrics.functional import peak_signal_noise_ratio as psnr

# ---------- Device ------------------
if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
print(f"Using device: {device}")

# ---------- Data --------------------
train_xs = torch.load("experiments/dilate/data/train_xs.pt")
train_ys = torch.load("experiments/dilate/data/train_ys.pt")

test_xs = torch.load("experiments/dilate/data/test_xs.pt")
test_ys = torch.load("experiments/dilate/data/test_ys.pt")

train_ds = TensorDataset(train_xs, train_ys)
test_ds = TensorDataset(test_xs, test_ys)

# ---------- Model -------------------
class ModelI(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Unflatten(1, (1,28,28)),
            nn.Conv2d(1,6,5,padding=2),
            nn.ReLU(),
            nn.Conv2d(6,6,5,padding=2),
            nn.ReLU(),
            nn.Conv2d(6,1,5,padding=2),
            nn.Flatten(1,2)
        )

    def forward(self, x):
        return self.net(x)

# ---------- Training ----------------
seeds = [0]
# batch_sizes = [8,32,128,512]
# batch_sizes = [64,256,512,1024]
batch_sizes = [64]
# learning_rates = [.01, .001, .0001, .00001]
learning_rates = [.001]
idxs = list(range(20))

# CNN measurements
test_loader = DataLoader(test_ds, batch_size=2048, shuffle=False)
# cnn = CNN().to(device)
# cnn_path = "experiments/dilate/data/mnist_cnn.pt"
# cnn.load_state_dict(torch.load(cnn_path, map_location=device))
# cnn.eval()

# PSNR measurements
# ones_test = torch.load("data/ones_test.pt")
# one_loader = DataLoader(ones_test, batch_size=6742, shuffle=False)
batch_psnr = torch.vmap(psnr)

loss_train = {}
loss_test = {}
one_test = {}
output_test = {}
psnr_test = {}
for seed in seeds:
    for batch_size in batch_sizes:
        for lr in learning_rates:   
            # Initialize random seed
            torch.manual_seed(seed)
            random.seed(seed)

            # Initialize data
            train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

            # Initialize model 
            modelI = ModelI().to(device)
            criterion = nn.L1Loss()
            optimizer = optim.Adam(modelI.parameters(), lr=lr)

            # Calculate initial measurements
            test_loss = 0.0
            # test_one = 0
            bpsnr = 0.0
            with torch.no_grad():
                # for data in one_loader:
                #     data = data.to(device)
                #     output = modelI(data)
                #     bpsnr = batch_psnr(output, data).mean().item()
                for data, target in test_loader:
                    data, target = data.to(device), target.to(device)

                    output = modelI(data)
                    bpsnr += batch_psnr(output, target).mean().item()
                    test_loss += criterion(output, target).item()

                    # isone = cnn(output).softmax(1)
                    # isone = isone[:,1].mean().item()
                    # test_one += isone

            test_loss /= len(test_loader)
            bpsnr /= len(test_loader)
            # test_one /= len(test_loader)
            loss_test[(0, seed, batch_size, lr)] = test_loss
            # one_test[(0, seed, batch_size, lr)] = test_one
            psnr_test[(0, seed, batch_size, lr)] = bpsnr

            for idx in idxs:
                x = test_ds[idx][0].unsqueeze(0).to(device)  # add batch dimension: (1,1,28,28)
                output = modelI(x).squeeze().tolist()
                output_test[(0, seed, batch_size, lr, idx)] = output
        
            step = 1
            freq = 100
            for epoch in range(30):
                print(f"Epoch: {epoch}, Lr: {lr}, Bs: {batch_size}, Seed: {seed}")

                for data, target in train_loader:
                    data, target = data.to(device), target.to(device)
                    optimizer.zero_grad()
                    output = modelI(data)
                    loss = criterion(output, target)
                    loss.backward()
                    optimizer.step()

                    # Record training information
                    loss_train[(step, seed, batch_size, lr)] = loss.item()
                    step += 1

                    if step % freq == 0:
                        if step >= 300:
                            freq = 100

                        test_loss = 0.0
                        test_one = 0
                        bpsnr = 0.0

                        with torch.no_grad():
                            for data, target in test_loader:
                                data, target = data.to(device), target.to(device)

                                output = modelI(data)
                                test_loss += criterion(output, target).item()
                                bpsnr += batch_psnr(output, target).mean().item()


                                # isone = cnn(output).softmax(1)
                                # isone = isone[:,1].mean().item()
                                # test_one += isone

                        # Record losses and one accuracy
                        test_loss /= len(test_loader)
                        bpsnr /= len(test_loader)
                        # test_one /= len(test_loader)
                        loss_test[(step, seed, batch_size, lr)] = test_loss
                        # one_test[(step, seed, batch_size, lr)] = test_one
                        psnr_test[(step, seed, batch_size, lr)] = bpsnr


                        # Record model outputs
                        for idx in idxs:
                            x = test_ds[idx][0].unsqueeze(0).to(device)  # add batch dimension: (1,1,28,28)
                            output = modelI(x).squeeze().tolist()
                            output_test[(0, seed, batch_size, lr, idx)] = output



# with open("experiments/dilate/data/indirect_loss_train.csv", "w", newline="") as f:
#     writer = csv.writer(f)
#     writer.writerow(["step", "seed", "batch size", "lr", "loss"])
#     for (step, seed, batch_size, lr), loss in loss_train.items():
#         writer.writerow([step, seed, batch_size, lr, loss])
# with open("experiments/dilate/data/indirect_loss_test.csv", "w", newline="") as f:
#     writer = csv.writer(f)
#     writer.writerow(["step", "seed", "batch size", "lr", "loss"])
#     for (step, seed, batch_size, lr), loss in loss_test.items():
#         writer.writerow([step, seed, batch_size, lr, loss])
# with open("experiments/dilate/data/indirect_output_test.csv", "w", newline="") as f:
#     writer = csv.writer(f)
#     writer.writerow(["step", "seed", "batch size", "lr", "idx", "output"])
#     for (step, seed, batch_size, lr, idx), output in output_test.items():
#         writer.writerow([step, seed, batch_size, lr, idx, output])
# with open("experiments/dilate/data/indirect_psnr_test.csv", "w", newline="") as f:
#     writer = csv.writer(f)
#     writer.writerow(["step", "seed", "batch size", "lr", "idx", "psnr"])
#     for (step, seed, batch_size, lr), snr in psnr_test.items():
#         writer.writerow([step, seed, batch_size, lr, idx, snr])


idxs = [8, 9, 10, 11, 12, 13, 14]

fig, axes = plt.subplots(nrows=7, ncols=2, figsize=(6, 8))
for row, idx in enumerate(idxs):
    # prepare input
    x = test_ds[idx][0]
    xim = x.squeeze().detach()
    with torch.no_grad():
        modelI.to('cpu')
        y = modelI(x.unsqueeze(0))
    yim = y.squeeze().detach()

    # plot input on the left, output on the right
    ax_in  = axes[row, 0]
    ax_out = axes[row, 1]

    ax_in.imshow(xim,  cmap="gray", vmin=0, vmax=1)
    ax_in.set_title(f"Input #{idx}")
    ax_in.axis("off")

    ax_out.imshow(yim, cmap="gray", vmin=0, vmax=1)
    ax_out.set_title(f"Output #{idx}")
    ax_out.axis("off")

plt.tight_layout()
plt.show()

