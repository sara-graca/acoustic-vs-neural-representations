import os
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from scipy.stats import spearmanr
import matplotlib.pyplot as plt
import seaborn as sns

os.makedirs("results/figures", exist_ok=True)
os.makedirs("results/tables", exist_ok=True)

df = pd.read_csv("data/features/features_acoustic_norm.csv")

# subsample for speed
N_SAMPLE = 500
np.random.seed(42)
idx = np.random.choice(len(df), N_SAMPLE, replace=False)
df_sub = df.iloc[idx].reset_index(drop=True)

# acoustic RSM: negative euclidean distance on F1_norm, F2_norm
ac_feats = df_sub[["F1_norm", "F2_norm"]].fillna(0).values
ac_dist  = cdist(ac_feats, ac_feats, metric="euclidean")
ac_rsm   = -ac_dist

MODELS = {
    "whisper_layer20": "data/features/features_whisper_layer20_pca.npz",
    "whisper_layer4":  "data/features/features_whisper_layer4_pca.npz",
    "xlsr_layer20":    "data/features/features_xlsr_layer20_pca.npz",
    "xlsr_layer10":    "data/features/features_xlsr_layer10_pca.npz",
    "xlsr_layer3":     "data/features/features_xlsr_layer3_pca.npz",
}

# compute neural RSMs
neural_rsms = {}
for model_name, npz_path in MODELS.items():
    data    = np.load(npz_path)
    X       = data["clustering"][idx]
    norms   = np.linalg.norm(X, axis=1, keepdims=True)
    X_norm  = X / np.maximum(norms, 1e-8)
    cos_sim = X_norm @ X_norm.T
    neural_rsms[model_name] = cos_sim

def mantel_test(rsm1, rsm2, n_perm=200):
    n = rsm1.shape[0]
    upper = np.triu_indices(n, k=1)
    v1 = rsm1[upper]
    v2 = rsm2[upper]
    observed_r, _ = spearmanr(v1, v2)
    perm_rs = []
    for _ in range(n_perm):
        perm_idx = np.random.permutation(n)
        rsm2_perm = rsm2[np.ix_(perm_idx, perm_idx)]
        r, _ = spearmanr(v1, rsm2_perm[upper])
        perm_rs.append(r)
    p_value = np.mean(np.array(perm_rs) >= abs(observed_r))
    return round(observed_r, 4), round(p_value, 4)

# Mantel tests: acoustic vs each neural
rows = []
for model_name, neural_rsm in neural_rsms.items():
    r, p = mantel_test(ac_rsm, neural_rsm)
    rows.append({
        "comparison":  f"acoustic vs {model_name}",
        "mantel_r":    r,
        "p_value":     p,
    })

# Mantel tests: neural vs neural
model_names = list(neural_rsms.keys())
for i, m1 in enumerate(model_names):
    for m2 in model_names[i+1:]:
        r, p = mantel_test(neural_rsms[m1], neural_rsms[m2])
        rows.append({
            "comparison": f"{m1} vs {m2}",
            "mantel_r":   r,
            "p_value":    p,
        })

mantel_df = pd.DataFrame(rows)
mantel_df.to_csv("results/tables/mantel_tests.csv", index=False)
print(mantel_df.to_string())