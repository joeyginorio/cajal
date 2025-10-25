import torch
import random
import csv
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets, transforms
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import numpy as np

bmidnight = (0, .329, .576)
bcayenne = (.8, 0, .14)

# df1 = pd.read_csv("experiments/dilate/data/indirect_loss_train.csv")
# df2 = pd.read_csv("experiments/dilate/data/indirect_loss_test.csv")
# df4 = pd.read_csv("experiments/dilate/data/indirect_output_test.csv")
# df5 = pd.read_csv("experiments/dilate/data/indirect_psnr_test.csv")

# idf1 = pd.read_csv("experiments/dilate/data/indirect_loss_train.csv")
# idf2 = pd.read_csv("experiments/dilate/data/indirect_loss_test.csv")
# idf4 = pd.read_csv("experiments/dilate/data/indirect_output_test.csv")
# idf5 = pd.read_csv("experiments/dilate/data/indirect_psnr_test.csv")

idxs = list(range(20))

# HYPERPARAMeters
lr = .001
bs = 32
xlim = 200

test_xs = torch.load("experiments/dilate/data/test_xs.pt")
test_ys = torch.load("experiments/dilate/data/test_ys.pt")
test_ds = TensorDataset(test_xs, test_ys)

direct_train_loss = df1[(df1["lr"] == lr) & (df1["batch size"] == bs)]
direct_test_loss = df2[(df2["lr"] == lr) & (df2["batch size"] == bs)]
direct_psnr = df5[(df5["lr"] == lr) & (df5["batch size"] == bs)]

indirect_train_loss = idf1[(idf1["lr"] == lr) & (idf1["batch size"] == bs)]
indirect_test_loss = idf2[(idf2["lr"] == lr) & (idf2["batch size"] == bs)]
indirect_psnr = idf5[(idf5["lr"] == lr) & (idf5["batch size"] == bs)]

def get_test_out(lr, batch_size, idx, df):
    test_out = []
    test_out.append(test_ds[idx][0].squeeze())
    test_out.append(eval(df[(df["lr"] == lr) & (df["batch size"] == batch_size) & (df["idx"] == idx) & (df["step"] == 0)].iloc[0].output))
    test_out.append(eval(df[(df["lr"] == lr) & (df["batch size"] == batch_size) & (df["idx"] == idx) & (df["step"] == 20)].iloc[0].output))
    test_out.append(eval(df[(df["lr"] == lr) & (df["batch size"] == batch_size) & (df["idx"] == idx) & (df["step"] == 40)].iloc[0].output))
    test_out.append(eval(df[(df["lr"] == lr) & (df["batch size"] == batch_size) & (df["idx"] == idx) & (df["step"] == 80)].iloc[0].output))
    test_out.append(eval(df[(df["lr"] == lr) & (df["batch size"] == batch_size) & (df["idx"] == idx) & (df["step"] == 120)].iloc[0].output))
    test_out.append(eval(df[(df["lr"] == lr) & (df["batch size"] == batch_size) & (df["idx"] == idx) & (df["step"] == 140)].iloc[0].output))
    test_out.append(eval(df[(df["lr"] == lr) & (df["batch size"] == batch_size) & (df["idx"] == idx) & (df["step"] == 160)].iloc[0].output))
    return test_out

idx1 = 10
idx2 = 9

direct_test_out = get_test_out(lr, bs, idx1, df4)
direct_test_out2 = get_test_out(lr, bs, idx2, df4)

indirect_test_out = get_test_out(lr, bs, idx1, idf4)
indirect_test_out2 = get_test_out(lr, bs, idx2, idf4)


sns.set_theme(style="white",     
              font="Futura",
              rc={                                      # global overrides
                "font.weight": "bold",
                "xtick.labelsize": 11,   # ← increase these numbers
                "ytick.labelsize": 11})

# Example: 5 vertically stacked plots that share the x-axis
fig, axes = plt.subplots(
    nrows=5, ncols=1,
    sharex=True,            # ← share the x-axis across all rows
    figsize=(6, 10),
    gridspec_kw={'height_ratios': [.9, .9, .9, .9, .9]}
)
axes[0].xaxis.label.set_fontsize(16) 
axes[4].set_xlabel("Steps", labelpad=12, fontweight="bold")   # default is ~4–6, so bump it up

# ------- DIRECT MODEL OUTPUT --------------
# x_coords = [0, 200, 400, 600, 800 ,1000, 1200, 1400]
# x_coords = [0, 40, 80, 120, 160 ,200, 240, 280]
x_coords = torch.linspace(0,xlim,8).tolist()
# set the x-ticks where you want them, and hide y-axis entirely
# axes[0].set_xticks(x_coords)
axes[0].set_yticks([])
axes[0].set_xlim(50, 550)   # give a little padding left/right
axes[0].set_ylim(-100,  130)   # images will be centered at y=0, so limit to about ±half-height

axes[0].axvline(x=1,           # x-position
           ymin=0.06, ymax=1, # relative 0–1 y-span (default is 0–1)
           color="black",
           linestyle=":",
           linewidth=2,
           label="Threshold")

# === place images ===
first = True
for x, img, img2 in zip(x_coords, direct_test_out, direct_test_out2):
    if first:
        # turn the array into an “OffsetImage”
        imbox = OffsetImage(img, zoom=1.2, cmap='gray')
        imbox2 = OffsetImage(img2, zoom=1.2, cmap='gray')
        # attach it at (x,0), centered
        ab = AnnotationBbox(imbox, (x, 80), frameon=True, box_alignment=(1.18,0.5),
                            bboxprops=dict(
                            facecolor=bmidnight,   # background fill
                            edgecolor="black",   # outline color
                            boxstyle="round,pad=0.25"))
        ab2 = AnnotationBbox(imbox2, (x, -40), frameon=True, box_alignment=(1.18,0.5),
                            bboxprops=dict(
                            facecolor=bmidnight,   # background fill
                            edgecolor="black",   # outline color
                            boxstyle="round,pad=0.25"))
        axes[0].add_artist(ab)
        axes[0].add_artist(ab2)
        first=False
        continue

    # turn the array into an “OffsetImage”
    imbox = OffsetImage(img, zoom=1.2, cmap='gray')
    imbox2 = OffsetImage(img2, zoom=1.2, cmap='gray')
    # attach it at (x,0), centered
    ab = AnnotationBbox(imbox, (x, 80), frameon=True, box_alignment=(1.1,0.5),
                        bboxprops=dict(
                         facecolor=bmidnight,   # background fill
                         edgecolor="black",   # outline color
                         boxstyle="round,pad=0.15"))
    ab2 = AnnotationBbox(imbox2, (x, -40), frameon=True, box_alignment=(1.1,0.5),
                         bboxprops=dict(
                         facecolor=bmidnight,   # background fill
                         edgecolor="black",   # outline color
                         boxstyle="round,pad=0.15"))
    axes[0].add_artist(ab)
    axes[0].add_artist(ab2)
for spine in axes[0].spines.values():
    spine.set_visible(False)
axes[0].set_ylabel("Output (D)", labelpad=-30, fontsize=16, fontweight="bold")
axes[0].yaxis.set_label_coords(-.17, 0.55)
# axes[0].axhline(
#     y=-100,             # vertical position of the bar
#     xmin=0.005,          # 0.0 = left of axes, 1.0 = right of axes
#     xmax=.98,
#     color='black',
#     linewidth=1.5,
#     solid_capstyle='butt',
#     zorder=2           # make sure it sits above the axes background
# )
axes[0].plot(
    [0.015, .99],            # x start/end in axes fraction
    [-0.09, -0.09],           # y start/end in axes fraction
    transform=axes[0].transAxes,  # interpret coords in axes space
    clip_on=False,            # allow drawing outside the axes box
    color='black',
    linewidth=1.5,
    solid_capstyle='butt',
    zorder=2
)

# ------- INDIRECT MODEL OUTPUT --------------
x_coords = torch.linspace(0,xlim,8).tolist()

# set the x-ticks where you want them, and hide y-axis entirely
# axes[1].set_xticks(x_coords)
axes[1].set_yticks([])
axes[1].set_xlim(-50, 550)   # give a little padding left/right
axes[1].set_ylim(-40,  130)   # images will be centered at y=0, so limit to about ±half-height

axes[1].axvline(x=1,           # x-position
           ymin=0.03, ymax=.975, # relative 0–1 y-span (default is 0–1)
           color="black",
           linestyle=":",
           linewidth=2,
           label="Threshold")

first=True
for x, img, img2 in zip(x_coords, indirect_test_out, indirect_test_out2):
    if first:
        # turn the array into an “OffsetImage”
        imbox = OffsetImage(img, zoom=1.2, cmap='gray')
        imbox2 = OffsetImage(img2, zoom=1.2, cmap='gray')
        # attach it at (x,0), centered
        ab = AnnotationBbox(imbox, (x, 90), frameon=True, box_alignment=(1.18,0.5),
                            bboxprops=dict(
                            facecolor=bcayenne,   # background fill
                            edgecolor="black",   # outline color
                            boxstyle="round,pad=0.25"))
        ab2 = AnnotationBbox(imbox2, (x, 0), frameon=True, box_alignment=(1.18,0.5),
                            bboxprops=dict(
                            facecolor=bcayenne,   # background fill
                            edgecolor="black",   # outline color
                            boxstyle="round,pad=0.25"))
        axes[1].add_artist(ab)
        axes[1].add_artist(ab2) 
        first=False
        continue


    # turn the array into an “OffsetImage”
    imbox = OffsetImage(img, zoom=1.2, cmap='gray')
    imbox2 = OffsetImage(img2, zoom=1.2, cmap='gray')
    # attach it at (x,0), centered
    ab = AnnotationBbox(imbox, (x, 90), frameon=True, box_alignment=(1.1,0.5),
                        bboxprops=dict(
                         facecolor=bcayenne,   # background fill
                         edgecolor="black",   # outline color
                         boxstyle="round, pad=0.15"))
    ab2 = AnnotationBbox(imbox2, (x, 0), frameon=True, box_alignment=(1.1,0.5),
                         bboxprops=dict(
                         facecolor=bcayenne,   # background fill
                         edgecolor="black",   # outline color
                         boxstyle="round, pad=0.15"))
    axes[1].add_artist(ab)
    axes[1].add_artist(ab2)
for spine in axes[1].spines.values():
    spine.set_visible(False)
axes[1].set_ylabel("Output (I)", labelpad=-20, fontsize=16, fontweight="bold")
axes[1].yaxis.set_label_coords(-0.17, 0.50)



# TRAIN LOSS PLOT
sns.lineplot(
    data=direct_train_loss,
    x="step",
    y="loss",
    linestyle=':',    # ← dashed
    linewidth=1,       # optional thickness
    estimator="mean",      # collapse runs that share (step, batch_size)
    errorbar="sd",         # shaded ribbon = mean ± 1 standard deviation
    color=bmidnight,
    label="Direct",
    ax=axes[2]
    )
sns.lineplot(
    data=indirect_train_loss,
    x="step",
    y="loss",
    linestyle=':',    # ← dashed
    linewidth=1,       # optional thickness
    estimator="mean",      # collapse runs that share (step, batch_size)
    errorbar="sd",         # shaded ribbon = mean ± 1 standard deviation
    color=bcayenne,
    label="Indirect",
    ax=axes[2]
    )
# axes[2].set_ylim(0.04, .095)          # tidy bounds

axes[2].legend(loc=1,fontsize=12)
axes[2].set_xlim(0, xlim)           # show just the first 500 updates
axes[2].set(
    xlabel="Steps",
    ylabel="Train Loss"
)
axes[2].yaxis.label.set_fontsize(16)
axes[2].yaxis.label.set_weight("bold")   


axes[2].tick_params(
    axis='y',
    which='major',    # major ticks
    length=4,         # tick length in points
    width=1
)

# TEST LOSS PLOT
sns.lineplot(
    data=direct_test_loss,
    x="step",
    y="loss",
    linestyle=':',    # ← dashed
    linewidth=1,       # optional thickness
    estimator="mean",      # collapse runs that share (step, batch_size)
    errorbar="sd",         # shaded ribbon = mean ± 1 standard deviation
    color=bmidnight,
    label="Direct",
    ax=axes[3]
    )
sns.lineplot(
    data=indirect_test_loss,
    x="step",
    y="loss",
    linestyle=':',    # ← dashed
    linewidth=1,       # optional thickness
    estimator="mean",      # collapse runs that share (step, batch_size)
    errorbar="sd",         # shaded ribbon = mean ± 1 standard deviation
    color=bcayenne,
    label="Indirect",
    ax=axes[3]
    )
# axes[3].set_ylim(0.04, .095)          # tidy bounds

axes[3].legend(loc=1,fontsize=12)
axes[3].set_xlim(0, xlim)           # show just the first 500 updates
axes[3].set(
    xlabel="Steps",
    ylabel="Test Loss"
)
axes[3].yaxis.label.set_fontsize(16)         
axes[3].yaxis.label.set_weight("bold")   

axes[3].tick_params(
    axis='y',
    which='major',    # major ticks
    length=4,         # tick length in points
    width=1
)


# ONE ACC PLOT
sns.lineplot(
    data=direct_psnr,
    x="step",
    y="psnr",
    linestyle=':',    # ← dashed
    linewidth=1,       # optional thickness
    estimator="mean",      # collapse runs that share (step, batch_size)
    errorbar="sd",         # shaded ribbon = mean ± 1 standard deviation
    color=bmidnight,
    label="Direct",
    ax=axes[4]
    )
sns.lineplot(
    data=indirect_psnr,
    x="step",
    y="psnr",
    linestyle=':',    # ← dashed
    linewidth=1,       # optional thickness
    estimator="mean",      # collapse runs that share (step, batch_size)
    errorbar="sd",         # shaded ribbon = mean ± 1 standard deviation
    color=bcayenne,
    label="Indirect",
    ax=axes[4]
    )
# axes[1].set_ylim(0.04, .10)          # tidy bounds
axes[4].legend(loc=1,fontsize=12)
axes[4].set_xlim(0, xlim)           # show just the first 500 updates
axes[4].set(
    xlabel="Steps",
    ylabel="PSNR"
)
axes[4].yaxis.label.set_fontsize(16)    
axes[4].xaxis.label.set_fontsize(16)    
axes[4].yaxis.label.set_weight("bold")   
axes[4].tick_params(
    axis='y',
    which='major',    # major ticks
    length=4,         # tick length in points
    width=1
)


fig.suptitle("Early learning dynamics", fontsize=20, x=.54, y=1., fontweight="bold")
# fig.tight_layout(pad=0.5)   # pad controls minimal spacing
fig.subplots_adjust(
    left=0.19,    # space from the left edge of the figure
    right=0.9,   # space from the right edge
    bottom=0.12,  # space from the bottom edge
    top=0.94,     # space from the top edge (below your suptitle)
    wspace=0.25,  # width between columns
    hspace=0.20   # height between rows
)
plt.show()

fig.savefig(
    "experiments/dilate/figures/early_dynamics.pdf",
    format="pdf",
    dpi=900,              # controls resolution of any raster elements
    bbox_inches="tight",  # trims excess whitespace
    pad_inches=0.02       # small padding around the figure
)


# ---------- Analyze -----------------

# modelD = modelD.to("cpu")

# idxs = [0, 3, 5, 200, 603]

# fig, axes = plt.subplots(nrows=5, ncols=2, figsize=(6, 6))

# for row, idx in enumerate(idxs):
#     # prepare input
#     x = test_ds[idx][0]
#     xim = x.squeeze().detach()
#     with torch.no_grad():
#         y = modelD(x)
#     yim = y.squeeze().detach()

#     # plot input on the left, output on the right
#     ax_in  = axes[row, 0]
#     ax_out = axes[row, 1]

#     ax_in.imshow(xim,  cmap="gray", vmin=0, vmax=1)
#     ax_in.set_title(f"Input #{idx}")
#     ax_in.axis("off")

#     ax_out.imshow(yim, cmap="gray", vmin=0, vmax=1)
#     ax_out.set_title(f"Output #{idx}")
#     ax_out.axis("off")

# plt.tight_layout()
# plt.show()