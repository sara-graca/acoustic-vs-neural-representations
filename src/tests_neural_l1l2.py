import os
import numpy as np
import pandas as pd
from statsmodels.stats.multitest import multipletests

os.makedirs("results/tables", exist_ok=True)

df = pd.read_csv("data/features/features_acoustic_norm.csv")
VOWELS = ["a", "e", "i", "o", "u", "y", "ø", "ɛ", "ɑ", "ə"]

MODELS = {
    "whisper_layer4":  "data/features/features_whisper_layer4_pca.npz",
    "xlsr_layer3":    "data/features/features_xlsr_layer3_pca.npz",
}

N_PERM = 5000
np.random.seed(42)

rows = []

for model_name, npz_path in MODELS.items():
    data   = np.load(npz_path)
    X      = data["clustering"]
    norms  = np.linalg.norm(X, axis=1, keepdims=True)
    X_norm = X / np.maximum(norms, 1e-8)

    for phoneme in VOWELS:
        mask   = df["phoneme"].values == phoneme
        X_ph   = X_norm[mask]
        labels = df["l1_status"].values[mask]

        l1_idx = np.where(labels == "fr")[0]
        l2_idx = np.where(labels == "ru")[0]

        if len(l1_idx) < 3 or len(l2_idx) < 3:
            continue

        # observed centroid distance
        l1_centroid = X_ph[l1_idx].mean(axis=0)
        l2_centroid = X_ph[l2_idx].mean(axis=0)
        obs_dist    = 1 - (l1_centroid @ l2_centroid) / (
            np.linalg.norm(l1_centroid) * np.linalg.norm(l2_centroid) + 1e-8
        )

        # permutation test
        all_idx = np.concatenate([l1_idx, l2_idx])
        n_l1    = len(l1_idx)
        perm_dists = []

        for _ in range(N_PERM):
            perm = np.random.permutation(len(all_idx))
            perm_l1 = X_ph[all_idx[perm[:n_l1]]].mean(axis=0)
            perm_l2 = X_ph[all_idx[perm[n_l1:]]].mean(axis=0)
            d = 1 - (perm_l1 @ perm_l2) / (
                np.linalg.norm(perm_l1) * np.linalg.norm(perm_l2) + 1e-8
            )
            perm_dists.append(d)

        p_val = np.mean(np.array(perm_dists) >= obs_dist)

        rows.append({
            "model":    model_name,
            "phoneme":  phoneme,
            "obs_dist": round(obs_dist, 4),
            "p_raw":    round(p_val, 4),
        })

results = pd.DataFrame(rows)

for model_name in MODELS:
    mask = results["model"] == model_name
    _, p_adj, _, _ = multipletests(results.loc[mask, "p_raw"], method="fdr_bh")
    results.loc[mask, "p_adj"] = p_adj.round(4)

results["significant"] = results["p_adj"] < 0.05
results.to_csv("results/tables/l1_l2_neural_tests.csv", index=False)
print(results.to_string())
