import torch
import random
import csv
from cajal.compiling import compile, TypedTensor
import cajal.syntax as cj
from cajal.typing import check
from torch import nn, optim
from torch import vmap
from torch.utils.data import DataLoader, TensorDataset

# ---------- Device ------------------
if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
print(f"Using device: {device}")

# ---------- Data --------------------
train_xs1 = torch.load("experiments/and/data/train_and_xs1.pt")
train_xs2 = torch.load("experiments/and/data/train_and_xs2.pt")
train_ys = torch.load("experiments/and/data/train_and_ys.pt")

test_xs1 = torch.load("experiments/and/data/test_and_xs1.pt")
test_xs2 = torch.load("experiments/and/data/test_and_xs2.pt")
test_ys = torch.load("experiments/and/data/test_and_ys.pt")

train_ds = TensorDataset(train_xs1, train_xs2, train_ys)
test_ds = TensorDataset(test_xs1, test_xs2, test_ys)


cbool = cj.TyFun(cj.TyBool(), cj.TyFun(cj.TyBool(), cj.TyBool()))
cid = cj.TmFun('h',cbool, cj.TmVar('h'))
cnot = cj.TmFun('p', cbool,
             cj.TmFun('a', cj.TyBool(),
                   cj.TmFun('b', cj.TyBool(),
                        cj.TmApp(cj.TmApp(cj.TmVar('p'), cj.TmVar('b')), cj.TmVar('a')))))
cfalse = cj.TmFun('x', cj.TyBool(), 
               cj.TmFun('y', cj.TyBool(), 
                     cj.TmIf(cj.TmVar('x'), cj.TmVar('y'), cj.TmVar('y'))))
ctrue = cj.TmFun('x', cj.TyBool(), 
               cj.TmFun('y', cj.TyBool(), 
                     cj.TmIf(cj.TmVar('y'), cj.TmVar('x'), cj.TmVar('x'))))
cfls = cj.TmFun('g', cbool,
             cj.TmIf(cj.TmApp(cj.TmApp(cj.TmVar('g'),cj.TmTrue()),cj.TmFalse()),
                  cfalse,
                  cfalse))
cand = cj.TmFun('g', cbool,
             cj.TmIf(cj.TmApp(cj.TmApp(cj.TmVar('g'),cj.TmTrue()),cj.TmFalse()),
                  cid,
                  cfls))
coand = cj.TmApp(cj.TmApp(cj.TmApp(cj.TmApp(cand, cj.TmVar('f')), cj.TmVar('r')), cj.TmTrue()), cj.TmFalse())

# ---------- Model -------------------
class ModelD(nn.Module):
    def __init__(self):
        super().__init__()
        self.netx1 = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28 * 28, 400),
            nn.ReLU(),
            nn.Linear(400, 8)
        )
        self.netx2 = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28 * 28, 400),
            nn.ReLU(),
            nn.Linear(400, 8)
        )

        self.coand = coand
        check(self.coand, {'f': cbool, 'r': cbool})
        self.coand_compiled = vmap(compile(self.coand), 
                             in_dims=({'f': (TypedTensor(0, None)),
                                       'r': (TypedTensor(0, None))},),
                            out_dims=TypedTensor(0, None))

    def forward(self, x1, x2):
        env = {'f' : TypedTensor(self.netx1(x1).reshape(-1,4,2), cbool), 
               'r' : TypedTensor(self.netx2(x2).reshape(-1,4,2), cbool)}
        return self.coand_compiled(env).data

# ---------- Training ----------------
seeds = [0,1,2,3,4]
# seeds = [0]
batch_sizes = [64,128,256,512]
# batch_sizes = [512]
learning_rates = [.01, .001, .0001, .00001]
# learning_rates = [.001]

test_loader = DataLoader(test_ds, batch_size=2048, shuffle=False)
loss_train = {}
loss_test = {}
acc = {}
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
            criterion = nn.CrossEntropyLoss()
            optimizer = optim.Adam(modelD.parameters(), lr=lr)

            # Calculate initial measurements
            test_loss = 0.0
            correct = 0
            total = 0
            with torch.no_grad():
                for x1, x2, target in test_loader:
                    x1, x2, target = x1.to(device), x2.to(device), target.to(device)
                    output = modelD(x1, x2).squeeze(-1)
                    test_loss += criterion(output, target).item()

                    # Accuracy computation
                    preds = torch.argmax(output, dim=1)
                    correct += (preds == target).sum().item()
                    total += target.size(0)

            test_loss /= len(test_loader)
            accuracy = correct / total
            loss_test[(0, seed, batch_size, lr)] = test_loss
            acc[(0, seed, batch_size, lr)] = accuracy
        
            step = 1
            freq = 10
            for epoch in range(5):
                print(f"Epoch: {epoch}, Lr: {lr}, Bs: {batch_size}, Seed: {seed}")

                for x1, x2, target in train_loader:
                    x1, x2, target = x1.to(device), x2.to(device), target.to(device)
                    optimizer.zero_grad()
                    output = modelD(x1, x2).squeeze(-1)
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
                        correct = 0
                        total = 0
                        with torch.no_grad():
                            for x1, x2, target in test_loader:
                                x1, x2, target = x1.to(device), x2.to(device), target.to(device)
                                output = modelD(x1, x2).squeeze(-1)
                                test_loss += criterion(output, target).item()

                                # Accuracy computation
                                preds = torch.argmax(output, dim=1)
                                correct += (preds == target).sum().item()
                                total += target.size(0)

                        # Record losses and one accuracy
                        test_loss /= len(test_loader)
                        accuracy = correct / total
                        loss_test[(step, seed, batch_size, lr)] = test_loss
                        acc[(step, seed, batch_size, lr)] = accuracy



# with open("experiments/and/data/church_and_loss_train.csv", "w", newline="") as f:
#     writer = csv.writer(f)
#     writer.writerow(["step", "seed", "batch size", "lr", "loss"])
#     for (step, seed, batch_size, lr), loss in loss_train.items():
#         writer.writerow([step, seed, batch_size, lr, loss])
# with open("experiments/and/data/church_and_loss_test.csv", "w", newline="") as f:
#     writer = csv.writer(f)
#     writer.writerow(["step", "seed", "batch size", "lr", "loss"])
#     for (step, seed, batch_size, lr), loss in loss_test.items():
#         writer.writerow([step, seed, batch_size, lr, loss])
# with open("experiments/and/data/church_and_acc_test.csv", "w", newline="") as f:
#     writer = csv.writer(f)
#     writer.writerow(["step", "seed", "batch size", "lr", "acc"])
#     for (step, seed, batch_size, lr), acc in acc.items():
#         writer.writerow([step, seed, batch_size, lr, acc])