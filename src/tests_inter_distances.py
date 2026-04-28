import os
import numpy as np
import pandas as pd
from scipy.spatial.distance import mahalanobis
from scipy.stats import spearmanr
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import yaml

np.random.seed(42)

os.makedirs("results/tables", exist_ok=True)
os.makedirs("results/figures", exist_ok=True)

# Load data

df = pd.read_csv("data/features/features_acoustic_norm.csv")

with open("params.yaml", encoding="utf-8") as f:
    params = yaml.safe_load(f)["tests_inter_distances"]

VOWELS = params["vowels"]
N_BOOT = params["n_boot"]
PAIRS  = [tuple(p) for p in params["pairs"]]

df_v = (
    df[df["phoneme"].isin(VOWELS)]
    .dropna(subset=["F1_norm", "F2_norm"])
    .copy()
    .reset_index(drop=True)   
)

MODELS = {
    "whisper_layer4": "data/features/features_whisper_layer4_pca.npz",
    "xlsr_layer3":    "data/features/features_xlsr_layer3_pca.npz",
}
# Keep only paths that actually exist
MODELS = {k: v for k, v in MODELS.items() if os.path.exists(v)}

n_vowels = len(VOWELS)
speakers  = df_v["speaker_id"].unique()


# Acoustic distance matrices

ac_centroids = df_v.groupby("phoneme")[["F1_norm", "F2_norm"]].mean().loc[VOWELS]

# Pooled within-phoneme covariance for Mahalanobis
cov_list, weights = [], []
for p in VOWELS:
    sub = df_v[df_v["phoneme"] == p][["F1_norm", "F2_norm"]].dropna()
    if len(sub) > 2:
        cov_list.append(np.cov(sub.values.T))
        weights.append(len(sub) - 1)

total_weight  = sum(weights)
pooled_cov    = sum(w * c for w, c in zip(weights, cov_list)) / total_weight
pooled_cov_inv = np.linalg.inv(pooled_cov)

D_euclidean   = np.zeros((n_vowels, n_vowels))
D_mahalanobis = np.zeros((n_vowels, n_vowels))

for i, p1 in enumerate(VOWELS):
    for j, p2 in enumerate(VOWELS):
        v1 = ac_centroids.loc[p1].values
        v2 = ac_centroids.loc[p2].values
        D_euclidean[i, j]   = np.linalg.norm(v1 - v2)
        D_mahalanobis[i, j] = mahalanobis(v1, v2, pooled_cov_inv)


# Neural distance matrices

def load_neural_vowels(npz_path: str) -> np.ndarray:

    data  = np.load(npz_path)
    X_all = data["clustering"]                       # shape (len(df), d)
    # df_v was built from df; its original positions are in df_v.index
    X_v   = X_all[df_v.index.values]                # select vowel rows
    norms = np.linalg.norm(X_v, axis=1, keepdims=True)
    return X_v / np.maximum(norms, 1e-8)             # L2-normalise

neural_data       = {}   # model → L2-normed vowel array (len(df_v), d)
neural_dists      = {}   # model → cosine distance matrix (n_vowels, n_vowels)
neural_centroids  = {}   # model → {phoneme: centroid vector}

for model_name, npz_path in MODELS.items():
    X_norm = load_neural_vowels(npz_path)
    neural_data[model_name] = X_norm

    phoneme_labels = df_v["phoneme"].values
    centroids = {
        p: X_norm[phoneme_labels == p].mean(axis=0)
        for p in VOWELS
        if (phoneme_labels == p).sum() > 0
    }
    neural_centroids[model_name] = centroids

    D = np.zeros((n_vowels, n_vowels))
    for i, p1 in enumerate(VOWELS):
        for j, p2 in enumerate(VOWELS):
            c1, c2  = centroids[p1], centroids[p2]
            cos_sim = (c1 @ c2) / (np.linalg.norm(c1) * np.linalg.norm(c2) + 1e-8)
            D[i, j] = 1.0 - cos_sim
    neural_dists[model_name] = D

# Mantel tests

def mantel(D1: np.ndarray, D2: np.ndarray, n_perm: int = 1000) -> tuple[float, float]:
    """Mantel test: Spearman r on upper-triangular elements, permuted rows/cols."""
    upper      = np.triu_indices(D1.shape[0], k=1)
    v1, v2_obs = D1[upper], D2[upper]
    r_obs, _   = spearmanr(v1, v2_obs)
    perm_rs    = []
    for _ in range(n_perm):
        idx  = np.random.permutation(D2.shape[0])
        D2p  = D2[np.ix_(idx, idx)]
        r, _ = spearmanr(v1, D2p[upper])
        perm_rs.append(r)
    p_val = np.mean(np.abs(perm_rs) >= abs(r_obs))
    return round(float(r_obs), 4), round(float(p_val), 4)

mantel_rows = []

# Acoustic vs neural
for model_name, D_neural in neural_dists.items():
    for dist_name, D_ac in [("euclidean", D_euclidean),
                             ("mahalanobis", D_mahalanobis)]:
        r, p = mantel(D_ac, D_neural)
        mantel_rows.append({
            "comparison": f"{dist_name} vs {model_name}",
            "r": r, "p": p,
        })

# Neural vs neural (all pairs)
model_names = list(neural_dists.keys())
for i in range(len(model_names)):
    for j in range(i + 1, len(model_names)):
        m1, m2 = model_names[i], model_names[j]
        r, p = mantel(neural_dists[m1], neural_dists[m2])
        mantel_rows.append({"comparison": f"{m1} vs {m2}", "r": r, "p": p})

mantel_df = pd.DataFrame(mantel_rows)
mantel_df.to_csv("results/tables/distance_mantel.csv", index=False)
print("✓ Saved Mantel tests →", "results/tables/distance_mantel.csv")
print(mantel_df.to_string(index=False))


# Bootstrap CIs ALL representation types

# helpers

def safe_centroid(df_sub: pd.DataFrame, p: str, cols: list) -> np.ndarray | None:
    sub = df_sub[df_sub["phoneme"] == p][cols]
    if len(sub) < 2 or sub.isnull().all(axis=None):
        return None
    c = sub.mean().values
    return None if np.any(np.isnan(c)) else c


def acoustic_dist(df_sub: pd.DataFrame, p1: str, p2: str,
                  kind: str = "euclidean",
                  cov_inv: np.ndarray | None = None) -> float:
    c1 = safe_centroid(df_sub, p1, ["F1_norm", "F2_norm"])
    c2 = safe_centroid(df_sub, p2, ["F1_norm", "F2_norm"])
    if c1 is None or c2 is None:
        return np.nan
    if kind == "euclidean":
        return float(np.linalg.norm(c1 - c2))
    else:  # mahalanobis
        try:
            return float(mahalanobis(c1, c2, cov_inv))
        except Exception:
            return np.nan


def neural_dist(X_norm: np.ndarray, labels: np.ndarray,
                p1: str, p2: str) -> float:
    m1 = X_norm[labels == p1]
    m2 = X_norm[labels == p2]
    if len(m1) == 0 or len(m2) == 0:
        return np.nan
    c1, c2  = m1.mean(axis=0), m2.mean(axis=0)
    cos_sim = (c1 @ c2) / (np.linalg.norm(c1) * np.linalg.norm(c2) + 1e-8)
    return float(1.0 - cos_sim)


def bootstrap_ci(obs_val: float, boot_vals: list,
                 alpha: float = 0.05) -> tuple[float, float, float]:
    valid = [v for v in boot_vals if not np.isnan(v)]
    if len(valid) < 10:
        return obs_val, np.nan, np.nan
    return (
        round(obs_val, 4),
        round(float(np.percentile(valid, 100 * alpha / 2)), 4),
        round(float(np.percentile(valid, 100 * (1 - alpha / 2))), 4),
    )

# define all (representation, distance-function) combos

representations: list[tuple[str, callable]] = [
    ("acoustic_euclidean",
     lambda df_sub, p1, p2: acoustic_dist(df_sub, p1, p2, "euclidean")),
    ("acoustic_mahalanobis",
     lambda df_sub, p1, p2: acoustic_dist(df_sub, p1, p2, "mahalanobis",
                                           cov_inv=pooled_cov_inv)),
]

def make_neural_dist_fn(X_norm_captured):
    def fn(df_sub, p1, p2):
        mask = df_v.index.isin(df_sub.index)
        return neural_dist(
            X_norm_captured[mask],
            df_v.loc[mask, "phoneme"].values,
            p1, p2,
        )
    return fn

for model_name, X_norm in neural_data.items():
    representations.append((
        f"neural_{model_name}",
        make_neural_dist_fn(X_norm),
    ))

#run bootstrap

print(f"\nRunning {N_BOOT} bootstrap resamples × "
      f"{len(PAIRS)} pairs × {len(representations)} representations …")

boot_rows = []

for rep_name, dist_fn in representations:
    print(f"  {rep_name}")
    for p1, p2 in tqdm(PAIRS, leave=False):
        # observed value on full data
        obs = dist_fn(df_v, p1, p2)

        # bootstrap
        boot_vals = []
        for _ in range(N_BOOT):
            spk_sample  = np.random.choice(speakers, size=len(speakers), replace=True)
            df_boot     = pd.concat(
                [df_v[df_v["speaker_id"] == s] for s in spk_sample],
                ignore_index=False,          # keep original index for npz alignment
            )
            d = dist_fn(df_boot, p1, p2)
            boot_vals.append(d)

        obs_out, ci_lo, ci_hi = bootstrap_ci(obs, boot_vals)
        boot_rows.append({
            "representation": rep_name,
            "pair":           f"/{p1}/–/{p2}/",
            "obs":            obs_out,
            "ci_low":         ci_lo,
            "ci_high":        ci_hi,
        })

boot_df = pd.DataFrame(boot_rows)
boot_df.to_csv("results/tables/distance_bootstrap_ci.csv", index=False)
print("Saved bootstrap CIs:", "results/tables/distance_bootstrap_ci.csv")


# Visualisations

# Distance-matrix heatmaps 

all_matrices = (
    [("Acoustic\nEuclidean", D_euclidean),
     ("Acoustic\nMahalanobis", D_mahalanobis)]
    + [(m.replace("_", "\n"), D) for m, D in neural_dists.items()]
)
n_plots = len(all_matrices)
fig, axes = plt.subplots(1, n_plots, figsize=(5.5 * n_plots, 5.5))
if n_plots == 1:
    axes = [axes]

for ax, (title, D) in zip(axes, all_matrices):
    sns.heatmap(
        D, ax=ax,
        xticklabels=VOWELS, yticklabels=VOWELS,
        cmap="YlOrRd", annot=True, fmt=".2f",
        annot_kws={"size": 7},
        square=True,
    )
    ax.set_title(title, fontsize=10)

plt.suptitle("Pairwise phoneme distance matrices", fontsize=12, y=1.01)
plt.tight_layout()
plt.savefig("results/figures/distance_matrices.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved distance heatmaps:", "results/figures/distance_matrices.png")

# Bootstrap CI forest plots (one panel per representation)

pairs_labels = [f"/{p1}/–/{p2}/" for p1, p2 in PAIRS]
rep_names    = boot_df["representation"].unique()
n_reps       = len(rep_names)
fig, axes    = plt.subplots(1, n_reps, figsize=(4.5 * n_reps, 5), sharey=True)
if n_reps == 1:
    axes = [axes]

for ax, rep_name in zip(axes, rep_names):
    sub = boot_df[boot_df["representation"] == rep_name].set_index("pair")
    y_pos = np.arange(len(pairs_labels))

    for y, pair in zip(y_pos, pairs_labels):
        if pair not in sub.index:
            continue
        row = sub.loc[pair]
        ax.plot(row["obs"], y, "o", color="steelblue", zorder=3)
        ax.hlines(y, row["ci_low"], row["ci_high"],
                  color="steelblue", linewidth=2, zorder=2)

    ax.axvline(0, color="grey", linestyle="--", linewidth=0.8)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(pairs_labels, fontsize=9)
    ax.set_xlabel("Distance", fontsize=9)
    ax.set_title(rep_name.replace("_", "\n"), fontsize=9)
    ax.grid(axis="x", alpha=0.3)

plt.suptitle("Bootstrap 95% CIs on inter-phoneme distances\n"
             "(speaker-level resampling, B=2000)", fontsize=11)
plt.tight_layout()
plt.savefig("results/figures/distance_bootstrap_ci.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved CI forest plot:", "results/figures/distance_bootstrap_ci.png")