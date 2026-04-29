import os
import yaml
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from tqdm import tqdm
import statsmodels.formula.api as smf
from statsmodels.tools.sm_exceptions import ConvergenceWarning
from scipy.stats import chi2 as chi2_dist

warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

os.makedirs("results/tables", exist_ok=True)
os.makedirs("results/figures", exist_ok=True)


# 1. Parameters & data


with open("params.yaml", encoding="utf-8") as f:
    params = yaml.safe_load(f)

VOWELS = params["tests_inter_distances"]["vowels"]
N_BOOT  = params["ci_rope"]["n_boot"]
ROPE_HZ = params["ci_rope"]["rope_hz"]

MODELS_NEURAL = {
    "whisper_layer4": "data/features/features_whisper_layer4_pca.npz",
    "xlsr_layer3":    "data/features/features_xlsr_layer3_pca.npz",
}


df = pd.read_csv("data/features/features_acoustic_norm.csv")
df_v = (
    df[df["phoneme"].isin(VOWELS)]
    .dropna(subset=["F1_norm", "F2_norm"])
    .copy()
)
original_index = df_v.index.values
df_v = df_v.reset_index(drop=True)

df_v["L2"]   = (df_v["l1_status"] == "ru").astype(int)
df_v["Male"] = (df_v["gender"] == "m").astype(int)

speakers = df_v["speaker_id"].unique()


# 2. Load neural arrays


def load_neural(npz_path: str) -> np.ndarray:
    data  = np.load(npz_path)
    X_all = data["clustering"]
    X_v   = X_all[original_index]
    norms = np.linalg.norm(X_v, axis=1, keepdims=True)
    return X_v / np.maximum(norms, 1e-8)

neural_arrays = {name: load_neural(path) for name, path in MODELS_NEURAL.items()}


# 3. Convert acoustic ROPE from Hz to normalised units

# Lobanov normalisation: z = (x - mu_spk) / sigma_spk
# A difference of ROPE_HZ Hz maps to ROPE_HZ / mean(sigma_spk) in z-units

speaker_f1_std = df_v.groupby("speaker_id")["F1"].std().dropna()
mean_sigma_f1  = speaker_f1_std.mean()
speaker_f2_std = df_v.groupby("speaker_id")["F2"].std().dropna()
mean_sigma_f2  = speaker_f2_std.mean()

ROPE_F1_NORM = ROPE_HZ / mean_sigma_f1
ROPE_F2_NORM = ROPE_HZ / mean_sigma_f2

print(f"Mean speaker F1 std: {mean_sigma_f1:.1f} Hz")
print(f"Acoustic ROPE in normalised F1 units: ±{ROPE_F1_NORM:.4f}")
print(f"Acoustic ROPE in normalised F2 units: ±{ROPE_F2_NORM:.4f}")


# 4. Section 8.1 — Profile likelihood CIs per vowel


def fit_per_vowel(phoneme: str, formant: str) -> dict | None:
    """
    Fit extended model for a single vowel/formant combination.
    Returns dict with coef, ci_low, ci_high for L2 parameter.
    Uses profile likelihood CIs via model.conf_int().
    """
    sub = df_v[df_v["phoneme"] == phoneme][
        ["speaker_id", "L2", "Male", formant]
    ].dropna()

    if len(sub) < 20 or sub["L2"].nunique() < 2:
        return None

    # Standardise for stability
    std = sub[formant].std()
    if std == 0:
        return None
    sub = sub.copy()
    sub[formant] = (sub[formant] - sub[formant].mean()) / std

    try:
        model = smf.mixedlm(
            f"{formant} ~ L2 + Male",
            data=sub,
            groups=sub["speaker_id"],
        ).fit(reml=False, method="powell")

        ci = model.conf_int()   # profile likelihood CIs
        coef   = float(model.params["L2"])
        ci_low  = float(ci.loc["L2", 0])
        ci_high = float(ci.loc["L2", 1])
        pval    = float(model.pvalues["L2"])

        # Convert back to original (normalised) scale
        coef    *= std
        ci_low  *= std
        ci_high *= std

        return {
            "phoneme":  phoneme,
            "formant":  formant,
            "coef":     round(coef, 4),
            "ci_low":   round(ci_low, 4),
            "ci_high":  round(ci_high, 4),
            "p":        round(pval, 4),
        }
    except Exception:
        return None


acoustic_ci_rows = []
print("\nFitting per-vowel acoustic models...")
for phoneme in tqdm(VOWELS):
    for formant in ["F1_norm", "F2_norm"]:
        result = fit_per_vowel(phoneme, formant)
        if result:
            acoustic_ci_rows.append(result)

acoustic_ci_df = pd.DataFrame(acoustic_ci_rows)
acoustic_ci_df.to_csv("results/tables/rope_acoustic_ci.csv", index=False)
print("✓ Acoustic CIs →", "results/tables/rope_acoustic_ci.csv")


# 5. Bootstrap CIs on neural L1/L2 cosine distance


def cosine_dist(c1: np.ndarray, c2: np.ndarray) -> float:
    n1, n2 = np.linalg.norm(c1), np.linalg.norm(c2)
    if n1 == 0 or n2 == 0:
        return np.nan
    return float(1.0 - (c1 @ c2) / (n1 * n2))


def l1l2_cosine_dist(X: np.ndarray, l1_status: np.ndarray,
                     phoneme_labels: np.ndarray, phoneme: str) -> float:
    mask = phoneme_labels == phoneme
    l1_mask = mask & (l1_status == "fr")
    l2_mask = mask & (l1_status == "ru")
    if l1_mask.sum() == 0 or l2_mask.sum() == 0:
        return np.nan
    c1 = X[l1_mask].mean(axis=0)
    c2 = X[l2_mask].mean(axis=0)
    return cosine_dist(c1, c2)


neural_ci_rows = []
print("\nBootstrapping neural L1/L2 cosine distances...")

for model_name, X in neural_arrays.items():
    print(f"  {model_name}")
    phoneme_labels = df_v["phoneme"].values
    l1_status      = df_v["l1_status"].values

    for phoneme in tqdm(VOWELS, leave=False):
        obs = l1l2_cosine_dist(X, l1_status, phoneme_labels, phoneme)

        boot_vals = []
        for _ in range(N_BOOT):
            spk_sample = np.random.choice(speakers, size=len(speakers), replace=True)
            mask = np.isin(df_v["speaker_id"].values, spk_sample)
            # Weight by how many times each speaker was drawn
            idx = np.concatenate([
                np.where(df_v["speaker_id"].values == s)[0]
                for s in spk_sample
            ])
            X_boot      = X[idx]
            l1_boot     = l1_status[idx]
            ph_boot     = phoneme_labels[idx]
            d = l1l2_cosine_dist(X_boot, l1_boot, ph_boot, phoneme)
            boot_vals.append(d)

        valid = [v for v in boot_vals if np.isfinite(v)]
        if len(valid) < 10:
            continue

        neural_ci_rows.append({
            "model":    model_name,
            "phoneme":  phoneme,
            "obs":      round(float(obs), 4),
            "ci_low":   round(float(np.percentile(valid, 2.5)), 4),
            "ci_high":  round(float(np.percentile(valid, 97.5)), 4),
        })

neural_ci_df = pd.DataFrame(neural_ci_rows)
neural_ci_df.to_csv("results/tables/rope_neural_ci.csv", index=False)
print("✓ Neural CIs →", "results/tables/rope_neural_ci.csv")


# 6. Compute delta_0 (neural ROPE)

# delta_0 = mean intra-speaker cosine distance:
# average cosine distance between tokens of the same phoneme
# produced by the same speaker

print("\nComputing delta_0 (mean intra-speaker cosine distance)...")

delta0 = {}
for model_name, X in neural_arrays.items():
    dists = []
    for spk in speakers:
        spk_mask = df_v["speaker_id"].values == spk
        for phoneme in VOWELS:
            ph_mask  = df_v["phoneme"].values == phoneme
            mask     = spk_mask & ph_mask
            X_sub    = X[mask]
            if len(X_sub) < 2:
                continue
            # Mean pairwise cosine distance within this speaker/phoneme cell
            sim_mat = X_sub @ X_sub.T
            n = len(X_sub)
            upper   = np.triu_indices(n, k=1)
            if len(upper[0]) == 0:
                continue
            mean_sim  = sim_mat[upper].mean()
            mean_dist = 1.0 - mean_sim
            dists.append(mean_dist)

    delta0[model_name] = round(float(np.mean(dists)), 4)
    print(f"  delta_0 ({model_name}): {delta0[model_name]}")

pd.DataFrame([
    {"model": m, "delta_0": d} for m, d in delta0.items()
]).to_csv("results/tables/rope_delta0.csv", index=False)
print("✓ Delta_0 →", "results/tables/rope_delta0.csv")


# 7. Section 8.4 — ROPE classification


def classify_rope(ci_low: float, ci_high: float,
                  rope_low: float, rope_high: float) -> str:
    if ci_high <= rope_high and ci_low >= rope_low:
        return "Equivalent"
    elif ci_low >= rope_high or ci_high <= rope_low:
        return "Non-equivalent"
    else:
        return "Indeterminate"


rope_rows = []

# Acoustic
for _, row in acoustic_ci_df.iterrows():
    rope = ROPE_F1_NORM if row["formant"] == "F1_norm" else ROPE_F2_NORM
    classification = classify_rope(row["ci_low"], row["ci_high"], -rope, rope)
    rope_rows.append({
        "representation": f"acoustic_{row['formant']}",
        "phoneme":        row["phoneme"],
        "obs":            row["coef"],
        "ci_low":         row["ci_low"],
        "ci_high":        row["ci_high"],
        "rope_low":       round(-rope, 4),
        "rope_high":      round(rope, 4),
        "classification": classification,
        "p":              row["p"],
    })

# Neural
for _, row in neural_ci_df.iterrows():
    d0   = delta0[row["model"]]
    classification = classify_rope(row["ci_low"], row["ci_high"], 0.0, d0)
    rope_rows.append({
        "representation": f"neural_{row['model']}",
        "phoneme":        row["phoneme"],
        "obs":            row["obs"],
        "ci_low":         row["ci_low"],
        "ci_high":        row["ci_high"],
        "rope_low":       0.0,
        "rope_high":      d0,
        "classification": classification,
        "p":              "",
    })

rope_df = pd.DataFrame(rope_rows)
rope_df.to_csv("results/tables/rope_classification.csv", index=False)
print("✓ ROPE classification →", "results/tables/rope_classification.csv")


# 8. Forest plots


COLOR_MAP = {
    "Equivalent":     "steelblue",
    "Non-equivalent": "tomato",
    "Indeterminate":  "goldenrod",
}

# 8a. Acoustic forest plot (F1 and F2 side by side)
fig, axes = plt.subplots(1, 2, figsize=(14, 7), sharey=True)

for ax, formant, rope in zip(
    axes,
    ["F1_norm", "F2_norm"],
    [ROPE_F1_NORM, ROPE_F2_NORM],
):
    sub = rope_df[rope_df["representation"] == f"acoustic_{formant}"]
    y_pos = np.arange(len(sub))

    for y, (_, row) in zip(y_pos, sub.iterrows()):
        c = COLOR_MAP[row["classification"]]
        ax.plot(row["obs"], y, "o", color=c, zorder=3)
        ax.hlines(y, row["ci_low"], row["ci_high"],
                  color=c, linewidth=2, zorder=2)

    ax.axvspan(-rope, rope, alpha=0.1, color="grey", label="ROPE")
    ax.axvline(0, color="grey", linestyle="--", linewidth=0.8)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(sub["phoneme"], fontsize=10)
    ax.set_xlabel("L2 coefficient (normalised units)")
    ax.set_title(formant.replace("_norm", ""))
    ax.grid(axis="x", alpha=0.3)

patches = [mpatches.Patch(color=c, label=l) for l, c in COLOR_MAP.items()]
patches.append(mpatches.Patch(color="grey", alpha=0.3, label="ROPE"))
axes[1].legend(handles=patches, fontsize=8)
plt.suptitle("Acoustic L1/L2 contrasts with profile likelihood 95\\% CIs",
             fontsize=12)
plt.tight_layout()
plt.savefig("results/figures/rope_acoustic_forest.png",
            dpi=150, bbox_inches="tight")
plt.close()
print("✓ Acoustic forest plot →", "results/figures/rope_acoustic_forest.png")

# 8b. Neural forest plot
fig, axes = plt.subplots(1, len(neural_arrays), figsize=(7 * len(neural_arrays), 7),
                         sharey=True)
if len(neural_arrays) == 1:
    axes = [axes]

for ax, model_name in zip(axes, neural_arrays):
    d0  = delta0[model_name]
    sub = rope_df[rope_df["representation"] == f"neural_{model_name}"]
    y_pos = np.arange(len(sub))

    for y, (_, row) in zip(y_pos, sub.iterrows()):
        c = COLOR_MAP[row["classification"]]
        ax.plot(row["obs"], y, "o", color=c, zorder=3)
        ax.hlines(y, row["ci_low"], row["ci_high"],
                  color=c, linewidth=2, zorder=2)

    ax.axvspan(0, d0, alpha=0.1, color="grey", label=f"ROPE [0, {d0}]")
    ax.axvline(0, color="grey", linestyle="--", linewidth=0.8)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(sub["phoneme"], fontsize=10)
    ax.set_xlabel("L1/L2 cosine distance")
    ax.set_title(model_name.replace("_", " "))
    ax.grid(axis="x", alpha=0.3)

patches = [mpatches.Patch(color=c, label=l) for l, c in COLOR_MAP.items()]
patches.append(mpatches.Patch(color="grey", alpha=0.3, label="ROPE"))
axes[-1].legend(handles=patches, fontsize=8)
plt.suptitle("Neural L1/L2 cosine distances with bootstrap 95\\% CIs\n"
             f"(speaker-level resampling, B={N_BOOT})", fontsize=12)
plt.tight_layout()
plt.savefig("results/figures/rope_neural_forest.png",
            dpi=150, bbox_inches="tight")
plt.close()
print("✓ Neural forest plot →", "results/figures/rope_neural_forest.png")


# 9. Summary counts


print("\nROPE classification summary:")
summary = rope_df.groupby(["representation", "classification"]).size().unstack(
    fill_value=0
)
print(summary.to_string())
summary.to_csv("results/tables/rope_summary_counts.csv")
print("✓ ROPE summary counts →", "results/tables/rope_summary_counts.csv")