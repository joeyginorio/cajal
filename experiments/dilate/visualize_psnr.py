
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

plt.rcParams['axes.unicode_minus'] = False
plt.rcParams.update({
    "pdf.fonttype": 42,
    "ps.fonttype": 42
})


df5 = pd.read_csv("experiments/dilate/data/direct_psnr_test.csv")
tdf5 = pd.read_csv("experiments/dilate/data/type_psnr_test3.csv")
idf5 = pd.read_csv("experiments/dilate/data/indirect_psnr_test.csv")

df5["model"]  = "D"
tdf5["model"] = "T"
idf5["model"] = "I"

bmidnight = (0, .329, .576)
bcayenne = (.8, 0, .14)

palette = {
    "D": bmidnight,
    "T": "olivedrab",
    "I": bcayenne
}
hue_order = ["D", "T", "I"]

sns.set_theme(style="white",
              font="Futura",
              rc={
                "font.weight": "bold",
                "xtick.labelsize": 15,
                "ytick.labelsize": 15})

# ── 2. concatenate ───────────────────────────────────
big = pd.concat([df5, tdf5, idf5], ignore_index=True)

g = sns.relplot(
    data=big,
    kind="line",
    x="step", y="psnr",
    col="batch size",
    row="lr",
    alpha=.7,
    hue="model",
    style="model", 
    style_order=hue_order,
    dashes=[[], (3, 2,1,2), (1, 1)],
    units="seed",
    hue_order=hue_order,
    palette=palette, 
    estimator=None,
    # errorbar=("pi", 95),
    facet_kws=dict(sharex="col", sharey=True),
    height=2.8, aspect=1.2
)

g.set(ylim=(13, 23))
# g.set(xlim=(20, None))

g.set_axis_labels("step", "PSNR")

first_ax = g.axes[0, 0]
handles_all, labels_all = first_ax.get_legend_handles_labels()

if g._legend is not None:
    g._legend.remove()

g.set_titles("")

for r, lr_val in enumerate(g.row_names):
    for c, bs_val in enumerate(g.col_names):
        ax = g.axes[r, c]

        ax.legend(
            handles_all, labels_all,
            loc="lower right",
            fontsize=13.5,
            frameon=True,
            title=None
        )

        ax.text(
            0.02, 0.98, f"(bs: {bs_val}, lr: {lr_val})",
            transform=ax.transAxes,
            fontsize=13.5,
            verticalalignment='top',
            horizontalalignment='left',
            fontweight="bold"
        )

g.set_axis_labels("", "")
g.fig.supxlabel("Steps",  y=-0.01, fontsize=24, fontweight="bold")
g.fig.supylabel("PSNR", x=-0.01, fontsize=24, fontweight="bold")
g.fig.suptitle("PSNR Dynamics", y=1.02, fontsize=28, fontweight="bold")

g.fig.savefig(
    "experiments/dilate/figures/psnr_global.pdf",
    format="pdf",
    dpi=900,              # controls resolution of any raster elements
    bbox_inches="tight",  # trims excess whitespace
    pad_inches=0.02       # small padding around the figure
)

plt.show()