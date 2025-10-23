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
import torch.nn.functional as F
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
class ModelD(nn.Module):
    def __init__(self):
        super().__init__()

        # input: Bx1x28x28
        self.determine_n = nn.Sequential(
            nn.Flatten(),
            nn.Unflatten(1, (1,28,28)),
            nn.Conv2d(1,8,5, padding=2),
            nn.ReLU(),
            nn.Conv2d(8,16,5,padding=2),
            nn.ReLU(),
            nn.MaxPool2d(3),
            nn.Flatten(),
            nn.Linear(16*9*9,10) # B x 10
        )

        self.determine_conv = nn.Sequential(
            nn.Conv2d(1,8,5, padding=2),
            nn.ReLU(),
            nn.Conv2d(8,16,5,padding=2),
            nn.ReLU(),
            nn.MaxPool2d(3),
            nn.Flatten(),
            nn.Linear(16*9*9,10)
        )
        
        self.conv = nn.Sequential(
            nn.Flatten(),
            nn.Unflatten(1, (1,28,28)),
            nn.Conv2d(1,1,5,padding=2),
            nn.Flatten(1,2) # B x 28 x 28
        )

        self.p = cj.TmIter(cj.TmVar('base'), 
                           'y', 
                           cj.TmApp(cj.TmVar('iterator'), cj.TmVar('y')),
                           cj.TmVar('num'))

        self.p = torch.vmap(compile(self.p),
                            in_dims=({'base': (TypedTensor(0, None)),
                                      'iterator': None,
                                      'num': (TypedTensor(0, None))},),
                                      out_dims=TypedTensor(0, None))


    def forward(self, x):
        base_val = x # Bx28x28
        num_val = self.determine_n(x) # Bx10
        # iterator_weights = self.determine_conv(x)
        iterator_weights = torch.randn(1,1,5,5,device=x.device)
        iterator = lambda x : TypedTensor(F.conv2d(x.data.view(-1,1,28,28), 
                                                   iterator_weights,
                                                   padding=2),
                                          cj.TyBool())
        env = {'base' : TypedTensor(base_val, cj.TyBool()),
               'iterator' : iterator,
               'num' : TypedTensor(num_val, cj.TyNat())}
        return self.p(env).data.view(-1,28,28)

# def test19():
#     base_val = TypedTensor(torch.eye(10, device=device).unsqueeze(0), TyNat())
#     n_val = TypedTensor(torch.ones(10, device=device), TyNat())

#     tm = TmIter(TmVar('base'), 
#                 'y', 
#                 TmApp(TmVar('f'), TmVar('y')),
#                 TmVar('n'))
#     conv = torch.nn.Conv2d(1,1,
#                            kernel_size=7, 
#                            stride=1, 
#                            padding=3, 
#                            bias=False, 
#                            padding_mode="zeros")
#     conv.to(device)

#     def f_val(x):
#         return TypedTensor(conv(x.data), x.ty)
#     c_tm = compile(tm)
#     return c_tm({'f': f_val, 'base': base_val, 'n': n_val})

# ---------- Training ----------------
# seeds = [0]
seeds = [0]
# batch_sizes = [8,32,128,512]
# batch_sizes = [64,256,512,1024]
batch_sizes = [64]
# learning_rates = [.01, .001, .0001, .00001]
learning_rates = [.001]
idxs = list(range(20))

# # CNN measurements
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
            modelD = ModelD().to(device)
            criterion = nn.L1Loss()
            optimizer = optim.Adam(modelD.parameters(), lr=lr)

            # Calculate initial measurements
            test_loss = 0.0
            # test_one = 0
            bpsnr = 0.0
            with torch.no_grad():
                # for data in one_loader:
                #     data = data.to(device)
                #     output = modelD(data)
                #     bpsnr = batch_psnr(output, data).mean().item()
                for data, target in test_loader:
                    data, target = data.to(device), target.to(device)

                    output = modelD(data)
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
                output = modelD(x).squeeze().tolist()
                output_test[(0, seed, batch_size, lr, idx)] = output
        
            step = 1
            freq = 100
            for epoch in range(2):
                print(f"Epoch: {epoch}, Lr: {lr}, Bs: {batch_size}, Seed: {seed}")

                for data, target in train_loader:
                    data, target = data.to(device), target.to(device)
                    optimizer.zero_grad()
                    output = modelD(data)
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

                                output = modelD(data)
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
                            output = modelD(x).squeeze().tolist()
                            output_test[(0, seed, batch_size, lr, idx)] = output



with open("experiments/dilate/data/direct_loss_train.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["step", "seed", "batch size", "lr", "loss"])
    for (step, seed, batch_size, lr), loss in loss_train.items():
        writer.writerow([step, seed, batch_size, lr, loss])
with open("experiments/dilate/data/direct_loss_test.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["step", "seed", "batch size", "lr", "loss"])
    for (step, seed, batch_size, lr), loss in loss_test.items():
        writer.writerow([step, seed, batch_size, lr, loss])
with open("experiments/dilate/data/direct_output_test.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["step", "seed", "batch size", "lr", "idx", "output"])
    for (step, seed, batch_size, lr, idx), output in output_test.items():
        writer.writerow([step, seed, batch_size, lr, idx, output])
with open("experiments/dilate/data/direct_psnr_test.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["step", "seed", "batch size", "lr", "idx", "psnr"])
    for (step, seed, batch_size, lr), snr in psnr_test.items():
        writer.writerow([step, seed, batch_size, lr, idx, snr])


idxs = [0, 11, 12, 3, 4, 5, 6]

fig, axes = plt.subplots(nrows=7, ncols=2, figsize=(6, 8))
for row, idx in enumerate(idxs):
    # prepare input
    x = test_ds[idx][0]
    xim = x.squeeze().detach()
    with torch.no_grad():
        modelD.to('cpu')
        y = modelD(x.unsqueeze(0))
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

