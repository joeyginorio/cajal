
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
plt.rcParams['axes.unicode_minus'] = False

df5 = pd.read_csv("experiments/and/data/direct_and_acc_test.csv")
# idf5 = pd.read_csv("experiments/and/data/indirect_and_acc_test.csv")
tdf5 = pd.read_csv("experiments/and/data/type_and_acc_test.csv")
rdf5 = pd.read_csv("experiments/and/data/church_and_acc_test.csv")

df5["model"]  = "D"
# idf5["model"] = "I"
rdf5["model"] = "C"
tdf5["model"] = "T"

bmidnight = (0, .329, .576)
bcayenne = (.8, 0, .14)
purple = (.5, 0, .5)
green = (0, 0.5, 0)

palette = {
    "D":   bmidnight,      # bmidnight
    # "I": bcayenne,       # bcayenne
    "C": purple,
    "T": green
}
hue_order = ["D", "C", "T"]

sns.set_theme(style="white",
              font="Futura",
              rc={
                "font.weight": "bold",
                "xtick.labelsize": 15,
                "ytick.labelsize": 15})

# ── 2. concatenate ───────────────────────────────────
big = pd.concat([df5, tdf5, rdf5], ignore_index=True)

g = sns.relplot(
    data=big,
    kind="line",
    x="step", y="acc",
    col="batch size",           # share x-axis within each batch-size column
    row="lr",
    hue="model",
    hue_order=hue_order,
    palette=palette, 
    estimator="mean",           # average over seeds
    ci="sd",
    facet_kws=dict(sharex="col", sharey=True),
    height=2.8, aspect=1.2
)

g.set_axis_labels("step", "Accuracy")


# pull out a master copy of the handles/labels ─────────────
first_ax = g.axes[0, 0]                              # upper-left panel
handles_all, labels_all = first_ax.get_legend_handles_labels()

# drop Seaborn’s single, grid-level legend ────────────────
if g._legend is not None:
    g._legend.remove()

# optional: blank the facet titles ─────────────────────────
g.set_titles("")          # comment out if you still want them

# stitch a custom legend into *every* subplot ─────────────
for r, lr_val in enumerate(g.row_names):             # rows = learning rates
    for c, bs_val in enumerate(g.col_names):         # cols = batch sizes
        ax = g.axes[r, c]

        # -- build context-rich labels like "Model-A  (bs=32, lr=1e-3)"
        # -- keep the legend labels clean (just model names)
        ax.legend(
            handles_all, labels_all,
            loc="lower right",
            fontsize=13.5,
            frameon=True,
            title=None
        )

        # -- manually add bs/lr label at top-left of subplot
        ax.text(
            0.02, 0.98, f"(bs: {bs_val}, lr: {lr_val})",
            transform=ax.transAxes,
            fontsize=13.5,
            verticalalignment='top',
            horizontalalignment='left',
            fontweight="bold"
        )

for ax_row in g.axes:
    for ax in ax_row:
        ax.set_ylim(0.7, 1.0)

g.set_axis_labels("", "")        # wipes the “step” / “PSNR” strings
g.fig.supxlabel("Steps",  y=-0.01, fontsize=24, fontweight="bold")      # global x-axis label
g.fig.supylabel("Accuracy", x=-0.01, fontsize=24, fontweight="bold")       # global y-axis label
g.fig.suptitle("AND Dynamics", y=1.02, fontsize=28, fontweight="bold")       # global y-axis label

g.fig.savefig(
    "experiments/and/figures/and_acc_global.pdf",
    format="pdf",
    dpi=900,              # controls resolution of any raster elements
    bbox_inches="tight",  # trims excess whitespace
    pad_inches=0.02       # small padding around the figure
)

plt.show()