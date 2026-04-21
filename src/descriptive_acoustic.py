import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.transforms as transforms
import seaborn as sns
from matplotlib.patches import Ellipse

os.makedirs("results/figures", exist_ok=True)
os.makedirs("results/tables", exist_ok=True)

df = pd.read_csv("data/features/features_acoustic_norm.csv")

VOWELS = ["a", "e", "i", "o", "u", "y", "ø", "ɛ", "ɑ", "ə"]
df_v = df[df["phoneme"].isin(VOWELS)].dropna(subset=["F1_norm", "F2_norm"]).copy()

GROUPS = {
    ("fr", "f"): ("L1 French F", "#e91e8c"),   # pink
    ("fr", "m"): ("L1 French M", "#1e90ff"),   # blue
    ("ru", "f"): ("L2 Russian F", "#FFA500"),  # yellow
    ("ru", "m"): ("L2 Russian M", "#2e8b57"),  # green
}


def confidence_ellipse(x, y, ax, n_std=2.0, **kwargs):
    if len(x) < 3:
        return
    cov = np.cov(x, y)
    pearson = cov[0, 1] / np.sqrt(cov[0, 0] * cov[1, 1])
    rx = np.sqrt(1 + pearson)
    ry = np.sqrt(1 - pearson)
    ellipse = Ellipse((0, 0), width=rx * 2, height=ry * 2, **kwargs)
    scale_x = np.sqrt(cov[0, 0]) * n_std
    scale_y = np.sqrt(cov[1, 1]) * n_std
    t = transforms.Affine2D() \
        .rotate_deg(45) \
        .scale(scale_x, scale_y) \
        .translate(np.mean(x), np.mean(y))
    ellipse.set_transform(t + ax.transData)
    ax.add_patch(ellipse)

# 1. vowel chart — one subplot per vowel
n_cols = 4
n_rows = int(np.ceil(len(VOWELS) / n_cols))
fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 10))
axes = axes.flatten()

for idx, phoneme in enumerate(VOWELS):
    ax = axes[idx]
    for (l1, gender), (label, color) in GROUPS.items():
        ph = df_v[(df_v["phoneme"] == phoneme) &
                  (df_v["l1_status"] == l1) &
                  (df_v["gender"] == gender)]
        if len(ph) < 3:
            continue
        confidence_ellipse(
            ph["F2_norm"].values, ph["F1_norm"].values, ax,
            n_std=2.0, alpha=0.2, facecolor=color, edgecolor=color
        )
        ax.plot(ph["F2_norm"].mean(), ph["F1_norm"].mean(), "o", color=color, markersize=6)
    ax.invert_xaxis()
    ax.invert_yaxis()
    ax.set_title(f"/{phoneme}/", fontsize=13)
    ax.set_xlabel("F2")
    ax.set_ylabel("F1")

# hide empty subplots
for idx in range(len(VOWELS), len(axes)):
    axes[idx].set_visible(False)

handles = [mpatches.Patch(color=c, label=l) for (_, _), (l, c) in GROUPS.items()]
fig.legend(handles=handles, loc="lower right", fontsize=10)
fig.suptitle("French oral vowels — per-group centroids with 95% confidence ellipses", fontsize=14)
plt.tight_layout()
plt.savefig("results/figures/vowel_chart.pdf")
plt.savefig("results/figures/vowel_chart.png", dpi=150)
print("Saved vowel chart")
plt.close()

# 2. box plots of F1 and F2 per phoneme stratified by L1 status and gender
df_v["group"] = df_v["l1_status"].str.upper() + " " + df_v["gender"].str.upper()

for formant in ["F1_norm", "F2_norm"]:
    fig, ax = plt.subplots(figsize=(14, 5))
    sns.boxplot(
        data=df_v, x="phoneme", y=formant,
        hue="group", order=VOWELS,
        palette={"FR F": "#e91e8c", "FR M": "#1e90ff",
         "RU F": "#FFA500", "RU M": "#2e8b57"},
        ax=ax
    )
    ax.set_title(f"{formant} per phoneme by group")
    ax.set_xlabel("Phoneme")
    ax.set_ylabel(formant)
    plt.tight_layout()
    plt.savefig(f"results/figures/boxplot_{formant}.pdf")
    plt.savefig(f"results/figures/boxplot_{formant}.png", dpi=150)
    print(f"Saved boxplot {formant}")
    plt.close()

# 3. intra-speaker variability across repetitions
top_vowels = df_v["phoneme"].value_counts().head(6).index.tolist()

intra = df_v[df_v["phoneme"].isin(top_vowels)].groupby(
    ["speaker_id", "phoneme", "l1_status"]
)[["F1_norm", "F2_norm"]].std().reset_index()

for formant in ["F1_norm", "F2_norm"]:
    fig, ax = plt.subplots(figsize=(12, 5))
    sns.violinplot(
        data=intra, x="phoneme", y=formant,
        hue="l1_status", split=True,
        palette={"fr": "#e91e8c", "ru": "#FFA500"},
        inner="box", ax=ax
    )
    ax.set_title(f"Intra-speaker variability across repetitions: {formant}")
    ax.set_xlabel("Phoneme")
    ax.set_ylabel(f"Std of {formant} across repetitions")
    plt.tight_layout()
    plt.savefig(f"results/figures/violin_{formant}.pdf")
    plt.savefig(f"results/figures/violin_{formant}.png", dpi=150)
    print(f"Saved violin {formant}")
    plt.close()

# 4. descriptive stats table
rows = []
for phoneme in VOWELS:
    for (l1, gender), (label, _) in GROUPS.items():
        sub = df_v[(df_v["phoneme"] == phoneme) &
                   (df_v["l1_status"] == l1) &
                   (df_v["gender"] == gender)]
        if len(sub) == 0:
            continue
        for formant in ["F1_norm", "F2_norm"]:
            vals = sub[formant].dropna()
            rows.append({
                "phoneme": phoneme,
                "group":   label,
                "formant": formant,
                "n":       len(vals),
                "mean":    round(vals.mean(), 3),
                "median":  round(vals.median(), 3),
                "sd":      round(vals.std(), 3),
                "iqr":     round(vals.quantile(0.75) - vals.quantile(0.25), 3),
                "cv":      round(vals.std() / abs(vals.mean()), 3) if vals.mean() != 0 else np.nan,
            })

stats_df = pd.DataFrame(rows)
stats_df.to_csv("results/tables/descriptive_acoustic.csv", index=False)
print("Saved descriptive stats table")
