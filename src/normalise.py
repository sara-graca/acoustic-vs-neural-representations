import os
import yaml
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

with open("params.yaml") as f:
    params = yaml.safe_load(f)

PCA_VIZ = params["normalise"]["pca_dims_viz"]
PCA_CLUSTERING = params["normalise"]["pca_dims_clustering"]

ACOUSTIC_IN = os.path.join("data", "features", "features_acoustic.csv")
WHISPER_IN = os.path.join("data", "features", "features_whisper.npz")
XLSR_IN = os.path.join("data", "features", "features_xlsr.npz")

ACOUSTIC_OUT = os.path.join("data", "features", "features_acoustic_norm.csv")
WHISPER_PCA_OUT = os.path.join("data", "features", "features_whisper_pca.npz")
XLSR_PCA_OUT = os.path.join("data", "features", "features_xlsr_pca.npz")

# Lobanov normalisation on F1 and F2
df = pd.read_csv(ACOUSTIC_IN)

for formant in ["F1", "F2"]:
    col_norm = f"{formant}_norm"
    df[col_norm] = np.nan
    for speaker, group in df.groupby("speaker_id"):
        vals = group[formant].dropna()
        if len(vals) < 2:
            continue
        mean = vals.mean()
        std  = vals.std()
        if std == 0:
            continue
        df.loc[group.index, col_norm] = (df.loc[group.index, formant] - mean) / std

df.to_csv(ACOUSTIC_OUT, index=False)
print(f"Acoustic normalised → {ACOUSTIC_OUT}")

# PCA on neural representations
def apply_pca(npz_path, out_path, n_viz, n_clust):
    data = np.load(npz_path)
    X    = np.stack([data[str(i)] for i in range(len(data.files))])

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    pca_viz   = PCA(n_components=n_viz)
    pca_clust = PCA(n_components=n_clust)

    X_viz   = pca_viz.fit_transform(X_scaled)
    X_clust = pca_clust.fit_transform(X_scaled)

    np.savez(out_path, viz=X_viz, clustering=X_clust)
    print(f"PCA done → {out_path} | viz: {X_viz.shape} | clustering: {X_clust.shape}")

apply_pca(WHISPER_IN, WHISPER_PCA_OUT, PCA_VIZ, PCA_CLUSTERING)
apply_pca(XLSR_IN, XLSR_PCA_OUT, PCA_VIZ, PCA_CLUSTERING)
apply_pca("data/features/features_whisper_layer4.npz", "data/features/features_whisper_layer4_pca.npz", PCA_VIZ, PCA_CLUSTERING)
apply_pca("data/features/features_xlsr_layer3.npz", "data/features/features_xlsr_layer3_pca.npz", PCA_VIZ, PCA_CLUSTERING)