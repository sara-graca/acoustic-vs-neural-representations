import os
import yaml
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from scipy.spatial.distance import pdist, squareform
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

os.makedirs("results/tables", exist_ok=True)
os.makedirs("results/figures", exist_ok=True)

np.random.seed(42)


# 1. Parameters & data


with open("params.yaml", encoding="utf-8") as f:
    params = yaml.safe_load(f)

VOWELS = params["tests_inter_distances"]["vowels"]

MODELS_NEURAL = {
    "whisper_layer4": "data/features/features_whisper_layer4_pca.npz",
    "xlsr_layer3":    "data/features/features_xlsr_layer3_pca.npz",
}
MODELS_NEURAL = {k: v for k, v in MODELS_NEURAL.items() if os.path.exists(v)}

df = pd.read_csv("data/features/features_acoustic_norm.csv")
df_v = (
    df[df["phoneme"].isin(VOWELS)]
    .dropna(subset=["F1_norm", "F2_norm"])
    .copy()
)
original_index = df_v.index.values
df_v = df_v.reset_index(drop=True)

speakers = df_v["speaker_id"].unique()

# Speaker metadata (one row per speaker)
speaker_meta = (
    df_v.groupby("speaker_id")[["l1_status", "gender"]]
    .first()
    .loc[speakers]
    .reset_index()
)


# 2. Load neural arrays


def load_neural(npz_path: str) -> np.ndarray:
    data  = np.load(npz_path)
    X_all = data["clustering"]
    X_v   = X_all[original_index]
    norms = np.linalg.norm(X_v, axis=1, keepdims=True)
    return X_v / np.maximum(norms, 1e-8)

neural_arrays = {name: load_neural(path) for name, path in MODELS_NEURAL.items()}


# 3. Build per-speaker vectors


def speaker_vectors_acoustic(df_sub: pd.DataFrame,
                              phones: list) -> tuple[np.ndarray, list]:
    """
    Each speaker is represented as the concatenation of their
    per-phoneme mean [F1_norm, F2_norm] vectors.
    Speakers with missing phonemes are dropped.
    """
    rows, valid_spk = [], []
    for spk in speakers:
        spk_df = df_sub[df_sub["speaker_id"] == spk]
        vec = []
        ok  = True
        for p in phones:
            sub = spk_df[spk_df["phoneme"] == p][["F1_norm", "F2_norm"]].dropna()
            if len(sub) == 0:
                ok = False
                break
            vec.extend(sub.mean().values.tolist())
        if ok:
            rows.append(vec)
            valid_spk.append(spk)
    return np.array(rows), valid_spk


def speaker_vectors_neural(X: np.ndarray,
                            df_sub: pd.DataFrame,
                            phones: list) -> tuple[np.ndarray, list]:
    """
    Each speaker is represented as the concatenation of their
    per-phoneme mean neural vectors.
    """
    labels     = df_sub["phoneme"].values
    spk_labels = df_sub["speaker_id"].values
    rows, valid_spk = [], []

    for spk in speakers:
        spk_mask = spk_labels == spk
        vec = []
        ok  = True
        for p in phones:
            mask = spk_mask & (labels == p)
            if mask.sum() == 0:
                ok = False
                break
            vec.append(X[mask].mean(axis=0))
        if ok:
            rows.append(np.concatenate(vec))
            valid_spk.append(spk)

    return np.array(rows), valid_spk


# 4. Clustering helpers


def ward_cluster(M: np.ndarray, metric: str = "euclidean") -> np.ndarray:
    if metric == "cosine":
        D = squareform(pdist(M, metric="cosine"))
        return linkage(squareform(D), method="average", optimal_ordering=True)
    return linkage(M, method="ward", optimal_ordering=True)


def best_k(M: np.ndarray, Z: np.ndarray,
           k_range: range = range(2, 6)) -> tuple[int, float]:
    best_k_, best_sil = 2, -1
    for k in k_range:
        labels = fcluster(Z, k, criterion="maxclust")
        if len(np.unique(labels)) < 2:
            continue
        sil = silhouette_score(M, labels)
        if sil > best_sil:
            best_sil, best_k_ = sil, k
    return best_k_, best_sil


def ari_eval(Z: np.ndarray, valid_spk: list,
             meta: pd.DataFrame, col: str, k: int) -> float:
    labels_pred = fcluster(Z, k, criterion="maxclust")
    meta_sub    = meta.set_index("speaker_id").loc[valid_spk, col]
    unique      = list(set(meta_sub))
    labels_true = [unique.index(v) for v in meta_sub]
    return round(adjusted_rand_score(labels_true, labels_pred), 4)


GENDER_COLORS   = {"f": "hotpink", "m": "steelblue"}
L1_COLORS       = {"fr": "green", "ru": "darkorange"}

def plot_speaker_dendrogram(Z, valid_spk, meta, col, title, out_path):
    meta_sub  = meta.set_index("speaker_id").loc[valid_spk, col]
    unique    = sorted(set(meta_sub))
    
    if col == "gender":
        color_map = GENDER_COLORS
    else:
        color_map = L1_COLORS

    fig, ax = plt.subplots(figsize=(max(10, len(valid_spk) * 0.5), 5))
    dendrogram(Z, labels=valid_spk, ax=ax, leaf_rotation=90,
               color_threshold=0, above_threshold_color="grey")

    for tick in ax.get_xticklabels():
        spk   = tick.get_text()
        group = meta_sub.get(spk, "unknown")
        tick.set_color(color_map.get(group, "black"))
        tick.set_fontsize(7)

    patches = [mpatches.Patch(color=color_map[g], label=g) for g in unique]
    ax.legend(handles=patches, fontsize=8)
    ax.set_title(title, fontsize=10)
    ax.set_ylabel("Distance")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()


# 5. Run speaker clustering


print("=== Section 9.3: Speaker clustering ===")

ari_rows = []

# ── 5a. Acoustic ──
M_ac, valid_ac = speaker_vectors_acoustic(df_v, VOWELS)
M_ac_scaled    = StandardScaler().fit_transform(M_ac)
Z_ac           = ward_cluster(M_ac_scaled, metric="euclidean")
k_ac, sil_ac   = best_k(M_ac_scaled, Z_ac)

ari_l1_ac  = ari_eval(Z_ac, valid_ac, speaker_meta, "l1_status", k=2)
ari_gen_ac = ari_eval(Z_ac, valid_ac, speaker_meta, "gender",    k=2)

print(f"Acoustic — best k={k_ac}, silhouette={sil_ac:.3f}")
print(f"  ARI L1/L2: {ari_l1_ac}, ARI gender: {ari_gen_ac}")

ari_rows.append({
    "representation": "acoustic",
    "best_k": k_ac, "silhouette": round(sil_ac, 4),
    "ARI_L1L2": ari_l1_ac, "ARI_gender": ari_gen_ac,
})

for col, label in [("l1_status", "L1L2"), ("gender", "gender")]:
    plot_speaker_dendrogram(
        Z_ac, valid_ac, speaker_meta, col,
        title=f"Speaker clustering — Acoustic ({col})",
        out_path=f"results/figures/cluster_speaker_acoustic_{label}.png",
    )

# ── 5b. Neural ──
for model_name, X in neural_arrays.items():
    M_ne, valid_ne = speaker_vectors_neural(X, df_v, VOWELS)
    Z_ne           = ward_cluster(M_ne, metric="cosine")
    k_ne, sil_ne   = best_k(M_ne, Z_ne)

    ari_l1_ne  = ari_eval(Z_ne, valid_ne, speaker_meta, "l1_status", k=2)
    ari_gen_ne = ari_eval(Z_ne, valid_ne, speaker_meta, "gender",    k=2)

    print(f"{model_name} — best k={k_ne}, silhouette={sil_ne:.3f}")
    print(f"  ARI L1/L2: {ari_l1_ne}, ARI gender: {ari_gen_ne}")

    ari_rows.append({
        "representation": model_name,
        "best_k": k_ne, "silhouette": round(sil_ne, 4),
        "ARI_L1L2": ari_l1_ne, "ARI_gender": ari_gen_ne,
    })

    for col, label in [("l1_status", "L1L2"), ("gender", "gender")]:
        plot_speaker_dendrogram(
            Z_ne, valid_ne, speaker_meta, col,
            title=f"Speaker clustering — {model_name} ({col})",
            out_path=f"results/figures/cluster_speaker_{model_name}_{label}.png",
        )


# 6. Save results


ari_df = pd.DataFrame(ari_rows)
ari_df.to_csv("results/tables/clustering_speaker_ari.csv", index=False)
print("\n✓ ARI summary →", "results/tables/clustering_speaker_ari.csv")
print(ari_df.to_string(index=False))