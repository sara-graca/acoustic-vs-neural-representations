import os
import yaml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from scipy.spatial.distance import pdist, squareform
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.preprocessing import StandardScaler

os.makedirs("results/tables", exist_ok=True)
os.makedirs("results/figures", exist_ok=True)

np.random.seed(42)

# 1. Parameters & data

with open("params.yaml", encoding="utf-8") as f:
    params = yaml.safe_load(f)

VOWELS = params["tests_inter_distances"]["vowels"]
N_TOP_CONSONANTS = params["clustering_vowels"]["n_top_consonants"]
EXCLUDE_APPROX   = {"w", "ʁ", "l"}   # approximants to exclude from V+C

FRONTBACK = {
    "i": "front", "e": "front", "\u025b": "front",
    "y": "front", "\u00f8": "front",
    "a": "central",
    "u": "back", "o": "back", "\u0251": "back", "\u0259": "back",
}

HEIGHT = {
    "i": "high",  "y": "high",  "u": "high",
    "e": "mid",   "\u00f8": "mid", "o": "mid", "\u0259": "mid",
    "\u025b": "low_mid", "\u0251": "low_mid",
    "a": "low",
}

MODELS_NEURAL = {
    "whisper_layer4": "data/features/features_whisper_layer4_pca.npz",
    "xlsr_layer3":    "data/features/features_xlsr_layer3_pca.npz",
}
MODELS_NEURAL = {k: v for k, v in MODELS_NEURAL.items() if os.path.exists(v)}

df = pd.read_csv("data/features/features_acoustic_norm.csv")

# Vowel subset
df_v = (
    df[df["phoneme"].isin(VOWELS)]
    .dropna(subset=["F1_norm", "F2_norm"])
    .copy()
)
original_index_v = df_v.index.values
df_v = df_v.reset_index(drop=True)

# Top N consonants excluding approximants
all_cons = df[~df["phoneme"].isin(VOWELS)].copy()
top_cons_all = (
    all_cons["phoneme"].value_counts()
    .head(N_TOP_CONSONANTS)
    .index.tolist()
)
top_cons = [p for p in top_cons_all if p not in EXCLUDE_APPROX]
print(f"Consonants (no approximants): {top_cons}")

df_c = (
    all_cons[all_cons["phoneme"].isin(top_cons)]
    .dropna(subset=["scg", "duration_ms"])
    .copy()
)
original_index_c = df_c.index.values
df_c = df_c.reset_index(drop=True)

PHONES_VC = VOWELS + top_cons
CV_PARTITION = {p: "vowel" for p in VOWELS}
CV_PARTITION.update({p: "consonant" for p in top_cons})

# 2. Load neural arrays

def load_neural(npz_path: str, orig_idx: np.ndarray) -> np.ndarray:
    data  = np.load(npz_path)
    X_all = data["clustering"]
    X_v   = X_all[orig_idx]
    norms = np.linalg.norm(X_v, axis=1, keepdims=True)
    return X_v / np.maximum(norms, 1e-8)

neural_vowel = {n: load_neural(p, original_index_v) for n, p in MODELS_NEURAL.items()}

original_index_vc = np.concatenate([original_index_v, original_index_c])
neural_vc = {n: load_neural(p, original_index_vc) for n, p in MODELS_NEURAL.items()}

# 3. Centroid helpers

def phoneme_centroids_acoustic(df_sub, phones, features):
    rows, valid = [], []
    for p in phones:
        sub = df_sub[df_sub["phoneme"] == p][features].dropna()
        if len(sub) < 2:
            continue
        rows.append(sub.mean().values)
        valid.append(p)
    return np.array(rows), valid


def phoneme_centroids_neural(X, labels, phones):
    rows, valid = [], []
    for p in phones:
        mask = labels == p
        if mask.sum() < 2:
            continue
        rows.append(X[mask].mean(axis=0))
        valid.append(p)
    return np.array(rows), valid

# 4. Clustering helpers

def ward_cluster(centroids, metric="euclidean"):
    if metric == "cosine":
        D = squareform(pdist(centroids, metric="cosine"))
        return linkage(squareform(D), method="average", optimal_ordering=True)
    return linkage(centroids, method="ward", optimal_ordering=True)


def best_k(centroids, Z, k_range=range(2, 6)):
    best_k_, best_sil = 2, -1
    for k in k_range:
        labels = fcluster(Z, k, criterion="maxclust")
        if len(np.unique(labels)) < 2:
            continue
        sil = silhouette_score(centroids, labels)
        if sil > best_sil:
            best_sil, best_k_ = sil, k
    return best_k_, best_sil


def ari_evaluation(Z, phones, partition, k):
    labels_pred = fcluster(Z, k, criterion="maxclust")
    labels_true = [partition.get(p, "unknown") for p in phones]
    unique = list(set(labels_true))
    labels_true_int = [unique.index(l) for l in labels_true]
    return round(adjusted_rand_score(labels_true_int, labels_pred), 4)


def plot_dendrogram(Z, labels, title, out_path, color_partition=None):
    fig, ax = plt.subplots(figsize=(10, 5))
    # Disable scipy automatic branch colouring
    dendrogram(Z, labels=labels, ax=ax, leaf_rotation=0,
               orientation="top", color_threshold=0,
               above_threshold_color="grey")

    if color_partition:
        unique_groups = sorted(set(color_partition.values()))
        cmap      = plt.colormaps["tab10"].resampled(len(unique_groups))
        color_map = {g: cmap(i) for i, g in enumerate(unique_groups)}
        # Use tick.get_text() since dendrogram reorders labels
        for tick in ax.get_xticklabels():
            label = tick.get_text()
            group = color_partition.get(label, "unknown")
            tick.set_color(color_map.get(group, "black"))
        patches = [mpatches.Patch(color=color_map[g], label=g)
                   for g in unique_groups]
        ax.legend(handles=patches, fontsize=8)

    ax.set_title(title, fontsize=11)
    ax.set_ylabel("Distance")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()

# 5. Section 9.1 — Vowel clustering

print("\n=== Section 9.1: Vowel clustering ===")
ari_rows = []

# ── Acoustic F1+F2 ──
C_ac, phones_ac = phoneme_centroids_acoustic(df_v, VOWELS, ["F1_norm", "F2_norm"])
C_ac_scaled     = StandardScaler().fit_transform(C_ac)
Z_ac            = ward_cluster(C_ac_scaled, metric="euclidean")
k_ac, sil_ac    = best_k(C_ac_scaled, Z_ac)

ari_fb_ac = ari_evaluation(Z_ac, phones_ac, FRONTBACK, k=3)
ari_ht_ac = ari_evaluation(Z_ac, phones_ac, HEIGHT,    k=4)

print(f"Acoustic F1+F2 — best k={k_ac}, silhouette={sil_ac:.3f}")
print(f"  ARI front/back: {ari_fb_ac}, ARI height: {ari_ht_ac}")

ari_rows.append({
    "analysis": "vowel", "representation": "acoustic_F1F2",
    "best_k": k_ac, "silhouette": round(sil_ac, 4),
    "ARI_frontback": ari_fb_ac, "ARI_height": ari_ht_ac,
})

plot_dendrogram(Z_ac, phones_ac,
    title="Vowel clustering — Acoustic F1+F2 (Ward, Euclidean)",
    out_path="results/figures/cluster_vowel_acoustic.png",
    color_partition=FRONTBACK)

# ── Acoustic F1 only ──
C_ac_f1, phones_ac_f1 = phoneme_centroids_acoustic(df_v, VOWELS, ["F1_norm"])
C_ac_f1_scaled         = StandardScaler().fit_transform(C_ac_f1)
Z_ac_f1                = ward_cluster(C_ac_f1_scaled, metric="euclidean")
k_ac_f1, sil_ac_f1    = best_k(C_ac_f1_scaled, Z_ac_f1)

ari_fb_ac_f1 = ari_evaluation(Z_ac_f1, phones_ac_f1, FRONTBACK, k=3)
ari_ht_ac_f1 = ari_evaluation(Z_ac_f1, phones_ac_f1, HEIGHT,    k=4)

print(f"Acoustic F1 only — best k={k_ac_f1}, silhouette={sil_ac_f1:.3f}")
print(f"  ARI front/back: {ari_fb_ac_f1}, ARI height: {ari_ht_ac_f1}")

ari_rows.append({
    "analysis": "vowel", "representation": "acoustic_F1only",
    "best_k": k_ac_f1, "silhouette": round(sil_ac_f1, 4),
    "ARI_frontback": ari_fb_ac_f1, "ARI_height": ari_ht_ac_f1,
})

plot_dendrogram(Z_ac_f1, phones_ac_f1,
    title="Vowel clustering — Acoustic F1 only (Ward, Euclidean)",
    out_path="results/figures/cluster_vowel_acoustic_f1only.png",
    color_partition=HEIGHT)

# ── Neural ──
for model_name, X in neural_vowel.items():
    labels_v        = df_v["phoneme"].values
    C_ne, phones_ne = phoneme_centroids_neural(X, labels_v, VOWELS)
    Z_ne            = ward_cluster(C_ne, metric="cosine")
    k_ne, sil_ne    = best_k(C_ne, Z_ne)

    ari_fb_ne = ari_evaluation(Z_ne, phones_ne, FRONTBACK, k=3)
    ari_ht_ne = ari_evaluation(Z_ne, phones_ne, HEIGHT,    k=4)

    print(f"{model_name} — best k={k_ne}, silhouette={sil_ne:.3f}")
    print(f"  ARI front/back: {ari_fb_ne}, ARI height: {ari_ht_ne}")

    ari_rows.append({
        "analysis": "vowel", "representation": model_name,
        "best_k": k_ne, "silhouette": round(sil_ne, 4),
        "ARI_frontback": ari_fb_ne, "ARI_height": ari_ht_ne,
    })

    plot_dendrogram(Z_ne, phones_ne,
        title=f"Vowel clustering — {model_name} (average linkage, cosine)",
        out_path=f"results/figures/cluster_vowel_{model_name}.png",
        color_partition=FRONTBACK)

# 6. Section 9.2 — Vowel + consonant clustering

print("\n=== Section 9.2: Vowel + consonant clustering ===")

def build_vc_acoustic_centroids(df_v_sub, df_c_sub, vowels, consonants):
    rows, valid = [], []
    scg_mean = df_c_sub["scg"].mean()
    f1_mean  = df_v_sub["F1_norm"].mean()
    f2_mean  = df_v_sub["F2_norm"].mean()

    for p in vowels:
        sub = df_v_sub[df_v_sub["phoneme"] == p][
            ["F1_norm", "F2_norm", "duration_ms"]
        ].dropna()
        if len(sub) < 2:
            continue
        m = sub.mean().values
        rows.append(np.append(m, scg_mean))
        valid.append(p)

    for p in consonants:
        sub = df_c_sub[df_c_sub["phoneme"] == p][
            ["duration_ms", "scg"]
        ].dropna()
        if len(sub) < 2:
            continue
        m = sub.mean().values
        rows.append(np.array([f1_mean, f2_mean, m[0], m[1]]))
        valid.append(p)

    return np.array(rows), valid


# Acoustic V+C
C_vc_ac, phones_vc_ac = build_vc_acoustic_centroids(df_v, df_c, VOWELS, top_cons)
C_vc_ac_scaled         = StandardScaler().fit_transform(C_vc_ac)
Z_vc_ac                = ward_cluster(C_vc_ac_scaled, metric="euclidean")
k_vc_ac, sil_vc_ac    = best_k(C_vc_ac_scaled, Z_vc_ac, k_range=range(2, 8))
ari_cv_ac              = ari_evaluation(Z_vc_ac, phones_vc_ac, CV_PARTITION, k=2)

print(f"Acoustic V+C — best k={k_vc_ac}, silhouette={sil_vc_ac:.3f}")
print(f"  ARI C/V boundary: {ari_cv_ac}")

ari_rows.append({
    "analysis": "vowel+consonant", "representation": "acoustic",
    "best_k": k_vc_ac, "silhouette": round(sil_vc_ac, 4),
    "ARI_frontback": "", "ARI_height": "", "ARI_CV": ari_cv_ac,
})

plot_dendrogram(Z_vc_ac, phones_vc_ac,
    title="Vowel + Consonant clustering — Acoustic (Ward, Euclidean)",
    out_path="results/figures/cluster_vc_acoustic.png",
    color_partition=CV_PARTITION)

# Neural V+C
labels_vc = np.array(df_v["phoneme"].tolist() + df_c["phoneme"].tolist())

for model_name, X in neural_vc.items():
    C_vc_ne, phones_vc_ne = phoneme_centroids_neural(X, labels_vc, PHONES_VC)
    Z_vc_ne                = ward_cluster(C_vc_ne, metric="cosine")
    k_vc_ne, sil_vc_ne    = best_k(C_vc_ne, Z_vc_ne, k_range=range(2, 8))
    ari_cv_ne              = ari_evaluation(Z_vc_ne, phones_vc_ne, CV_PARTITION, k=2)

    print(f"{model_name} V+C — best k={k_vc_ne}, silhouette={sil_vc_ne:.3f}")
    print(f"  ARI C/V boundary: {ari_cv_ne}")

    ari_rows.append({
        "analysis": "vowel+consonant", "representation": model_name,
        "best_k": k_vc_ne, "silhouette": round(sil_vc_ne, 4),
        "ARI_frontback": "", "ARI_height": "", "ARI_CV": ari_cv_ne,
    })

    plot_dendrogram(Z_vc_ne, phones_vc_ne,
        title=f"Vowel + Consonant clustering — {model_name} (average, cosine)",
        out_path=f"results/figures/cluster_vc_{model_name}.png",
        color_partition=CV_PARTITION)

# 7. Save ARI summary

ari_df = pd.DataFrame(ari_rows)
ari_df.to_csv("results/tables/clustering_vowel_ari.csv", index=False)
print("\n✓ ARI summary →", "results/tables/clustering_vowel_ari.csv")
print(ari_df.to_string(index=False))