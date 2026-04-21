import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import umap

os.makedirs("results/figures", exist_ok=True)
os.makedirs("results/tables", exist_ok=True)

df = pd.read_csv("data/features/features_acoustic_norm.csv")

MODELS = {
    "whisper_layer20": "data/features/features_whisper_layer20_pca.npz",
    "whisper_layer4":  "data/features/features_whisper_layer4_pca.npz",
    "xlsr_layer20":    "data/features/features_xlsr_layer20_pca.npz",
    "xlsr_layer10":    "data/features/features_xlsr_layer10_pca.npz",
    "xlsr_layer3":     "data/features/features_xlsr_layer3_pca.npz",
}

all_phonemes = df["phoneme"].unique()
cmap = plt.cm.get_cmap("tab20", len(all_phonemes))
COLORS_PHONEME = {p: cmap(i) for i, p in enumerate(all_phonemes)}
COLORS_L1     = {"fr": "#e91e8c", "ru": "#FFA500"}
COLORS_GENDER = {"f": "#1e90ff", "m": "#2e8b57"}

def plot_2d(X, labels, colors_map, title, path, alpha=0.3, size=5, show_legend=True):
    fig, ax = plt.subplots(figsize=(9, 7))
    for label in sorted(set(labels)):
        mask = np.array(labels) == label
        ax.scatter(X[mask, 0], X[mask, 1],
                   color=colors_map.get(label, "gray"),
                   label=label, alpha=alpha, s=size)
    if show_legend:
        ax.legend(markerscale=3, fontsize=8)
    ax.set_title(title)
    ax.set_xlabel("Dim 1")
    ax.set_ylabel("Dim 2")
    plt.tight_layout()
    plt.savefig(path + ".png", dpi=150)
    plt.close()

variance_rows = []
similarity_rows = []

for model_name, npz_path in MODELS.items():
    data    = np.load(npz_path)
    X_clust = data["clustering"]  # 50-dim PCA
    X_viz   = data["viz"]         # 2-dim PCA

    reducer = umap.UMAP(n_components=2, random_state=42)
    X_umap  = reducer.fit_transform(X_clust)

    for method, X2d in [("pca", X_viz), ("umap", X_umap)]:
        for color_by, colors_map in [
            ("phoneme", COLORS_PHONEME),
            ("l1_status", COLORS_L1),
            ("gender", COLORS_GENDER),
        ]:
            labels = df[color_by].tolist()
            title  = f"{model_name} — {method} — coloured by {color_by}"
            path   = f"results/figures/neural_{model_name}_{method}_{color_by}"
            plot_2d(X2d, labels, colors_map, title, path, show_legend=(color_by != "phoneme"))

        # between-class variance ratio — all phonemes
        labels_all  = df["phoneme"].values
        total_var   = np.var(X2d, axis=0).sum()
        global_mean = X2d.mean(axis=0)
        centroids   = {p: X2d[labels_all == p].mean(axis=0)
                       for p in all_phonemes if (labels_all == p).sum() > 0}
        between_var = np.mean([
            ((c - global_mean) ** 2).sum()
            for c in centroids.values()
        ])
        variance_rows.append({
            "model":         model_name,
            "method":        method,
            "between_var":   round(between_var, 4),
            "total_var":     round(total_var, 4),
            "between_ratio": round(between_var / total_var, 4),
        })

    # cosine similarity within vs between phoneme — all phonemes, 50-dim
    labels_all = df["phoneme"].values
    norms      = np.linalg.norm(X_clust, axis=1, keepdims=True)
    X_norm     = X_clust / np.maximum(norms, 1e-8)

    within_sims, between_sims = [], []
    phoneme_list = [p for p in all_phonemes if (labels_all == p).sum() > 1]

    for p in phoneme_list:
        idx = np.where(labels_all == p)[0][:200]
        X_p = X_norm[idx]
        sim_matrix = X_p @ X_p.T
        upper = sim_matrix[np.triu_indices(len(idx), k=1)]
        within_sims.extend(upper.tolist())

    for i, p1 in enumerate(phoneme_list):
        for p2 in phoneme_list[i+1:]:
            idx1 = np.where(labels_all == p1)[0][:100]
            idx2 = np.where(labels_all == p2)[0][:100]
            sim  = (X_norm[idx1] @ X_norm[idx2].T).flatten()
            between_sims.extend(sim.tolist())

    similarity_rows.append({
        "model":      model_name,
        "within_sim": round(np.mean(within_sims), 4),
        "between_sim":round(np.mean(between_sims), 4),
        "ratio":      round(np.mean(within_sims) / np.mean(between_sims), 4),
    })

pd.DataFrame(variance_rows).to_csv("results/tables/neural_variance_ratio.csv", index=False)
pd.DataFrame(similarity_rows).to_csv("results/tables/neural_cosine_similarity.csv", index=False)
print("Saved variance and similarity tables")