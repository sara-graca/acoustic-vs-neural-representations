import os
import yaml
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from statsmodels.stats.contingency_tables import mcnemar
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

os.makedirs("results/tables", exist_ok=True)
os.makedirs("results/figures", exist_ok=True)

# 1. Parameters & data

with open("params.yaml", encoding="utf-8") as f:
    params = yaml.safe_load(f)

VOWELS = params["tests_inter_distances"]["vowels"]


MODELS = {
    "whisper_layer4":  "data/features/features_whisper_layer4_pca.npz",
    "xlsr_layer3":     "data/features/features_xlsr_layer3_pca.npz",
}
MODELS = {k: v for k, v in MODELS.items() if os.path.exists(v)}

# Load data 
df = pd.read_csv("data/features/features_acoustic_norm.csv")

df_v = (
    df[df["phoneme"].isin(VOWELS)]
    .dropna(subset=["F1_norm", "F2_norm"])
    .copy()
)
original_index = df_v.index.values   # save BEFORE reset
df_v = df_v.reset_index(drop=True)

speakers = df_v["speaker_id"].unique()

# Load neural arrays
def load_neural(npz_path: str) -> np.ndarray:
    data  = np.load(npz_path)
    X_all = data["clustering"]
    return X_all[original_index]

neural_arrays = {name: load_neural(path) for name, path in MODELS.items()}


# 3. Nearest-centroid classifier helpers


def fit_centroids_acoustic(df_train: pd.DataFrame) -> dict:
    return {
        p: df_train[df_train["phoneme"] == p][["F1_norm", "F2_norm"]].mean().values
        for p in VOWELS
        if (df_train["phoneme"] == p).sum() > 0
    }

def fit_centroids_neural(X_train: np.ndarray, labels_train: np.ndarray) -> dict:
    return {
        p: X_train[labels_train == p].mean(axis=0)
        for p in VOWELS
        if (labels_train == p).sum() > 0
    }

def predict_nearest(centroids: dict, X_test: np.ndarray,
                    metric: str = "euclidean") -> np.ndarray:
    classes  = list(centroids.keys())
    C        = np.stack([centroids[p] for p in classes])  # (n_classes, d)
    diffs    = X_test[:, None, :] - C[None, :, :]         # (n, n_classes, d)
    dists    = np.linalg.norm(diffs, axis=2)               # (n, n_classes)
    pred_idx = np.argmin(dists, axis=1)
    return np.array([classes[i] for i in pred_idx])


# 4. Leave-one-speaker-out CV


def loso_cv(rep_name: str) -> pd.DataFrame:
    """
    Run LOSO CV for one representation.
    Returns a DataFrame with columns:
        token_idx, true, pred, speaker_id, l1_status
    """
    rows = []
    for spk in tqdm(speakers, desc=rep_name, leave=False):
        test_mask  = df_v["speaker_id"] == spk
        train_mask = ~test_mask

        df_train = df_v[train_mask]
        df_test  = df_v[test_mask]

        true_labels = df_test["phoneme"].values

        if rep_name == "acoustic":
            centroids = fit_centroids_acoustic(df_train)
            X_test    = df_test[["F1_norm", "F2_norm"]].values
            preds     = predict_nearest(centroids, X_test, metric="euclidean")
        else:
            X_all     = neural_arrays[rep_name]
            X_train   = X_all[train_mask.values]
            X_test    = X_all[test_mask.values]
            labels_tr = df_train["phoneme"].values
            centroids = fit_centroids_neural(X_train, labels_tr)
            preds = predict_nearest(centroids, X_test, metric="euclidean")

        for idx, (t, p) in zip(df_test.index, zip(true_labels, preds)):
            rows.append({
                "token_idx":  idx,
                "true":       t,
                "pred":       p,
                "speaker_id": spk,
                "l1_status":  df_v.loc[idx, "l1_status"],
            })

    return pd.DataFrame(rows)

# Run for acoustic + all neural models
all_results = {}
all_results["acoustic"] = loso_cv("acoustic")
for model_name in neural_arrays:
    all_results[model_name] = loso_cv(model_name)


# 5. Accuracy, per-class F1, confusion matrices


summary_rows = []
for rep_name, res in all_results.items():
    acc    = accuracy_score(res["true"], res["pred"])
    f1_mac = f1_score(res["true"], res["pred"],
                      labels=VOWELS, average="macro", zero_division=0)
    f1_per = f1_score(res["true"], res["pred"],
                      labels=VOWELS, average=None,
                      zero_division=0)
    summary_rows.append({
        "representation": rep_name,
        "accuracy":       round(acc, 4),
        "macro_f1":       round(f1_mac, 4),
        **{f"f1_{p}": round(f, 4) for p, f in zip(VOWELS, f1_per)},
    })

    # Confusion matrix heatmap
    cm = confusion_matrix(res["true"], res["pred"], labels=VOWELS)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True).clip(min=1)

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm_norm, annot=True, fmt=".2f", cmap="Blues",
                xticklabels=VOWELS, yticklabels=VOWELS, ax=ax,
                vmin=0, vmax=1)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(f"Confusion matrix — {rep_name}\n"
                 f"acc={acc:.3f}, macro-F1={f1_mac:.3f}")
    plt.tight_layout()
    plt.savefig(f"results/figures/confusion_{rep_name}.png",
                dpi=150, bbox_inches="tight")
    plt.close()

summary_df = pd.DataFrame(summary_rows)
summary_df.to_csv("results/tables/classifier_summary.csv", index=False)
print("✓ Classifier summary →", "results/tables/classifier_summary.csv")
print(summary_df[["representation", "accuracy", "macro_f1"]].to_string(index=False))


# 6. McNemar tests


def mcnemar_test(res_a: pd.DataFrame, res_b: pd.DataFrame,
                 label: str) -> dict:
    """
    McNemar test on matched token pairs.
    Merges on token_idx so order doesn't matter.
    """
    merged = res_a[["token_idx", "true", "pred"]].merge(
    res_b[["token_idx", "pred"]],
    on="token_idx", suffixes=("_a", "_b"),
    )
    correct_a = merged["true"] == merged["pred_a"]
    correct_b = merged["true"] == merged["pred_b"]

    # Contingency table: [[both correct, a correct b wrong],
    #                      [a wrong b correct, both wrong]]
    n00 = ((~correct_a) & (~correct_b)).sum()   # both wrong
    n01 = ((~correct_a) &   correct_b ).sum()   # only b correct
    n10 = (  correct_a  & (~correct_b)).sum()   # only a correct
    n11 = (  correct_a  &   correct_b ).sum()   # both correct

    table = np.array([[n11, n10], [n01, n00]])
    result = mcnemar(table, exact=False, correction=True)

    return {
        "comparison": label,
        "n_tokens":   len(merged),
        "acc_a":      round(correct_a.mean(), 4),
        "acc_b":      round(correct_b.mean(), 4),
        "n10_a_only": int(n10),
        "n01_b_only": int(n01),
        "chi2":       round(float(result.statistic), 4),
        "p":          round(float(result.pvalue), 4),
    }

mcnemar_rows = []

# Across representation types
rep_names = list(all_results.keys())
for i in range(len(rep_names)):
    for j in range(i + 1, len(rep_names)):
        r = mcnemar_test(
            all_results[rep_names[i]],
            all_results[rep_names[j]],
            label=f"{rep_names[i]} vs {rep_names[j]}",
        )
        mcnemar_rows.append(r)

# L1 vs L2 subgroups within each representation
for rep_name, res in all_results.items():
    res_l1 = res[res["l1_status"] == "fr"]   # adjust value if your csv uses different labels
    res_l2 = res[res["l1_status"] == "ru"]

    acc_l1 = accuracy_score(res_l1["true"], res_l1["pred"])
    acc_l2 = accuracy_score(res_l2["true"], res_l2["pred"])

    mcnemar_rows.append({
        "comparison":  f"{rep_name} — L1 acc vs L2 acc",
        "n_tokens":    f"L1={len(res_l1)}, L2={len(res_l2)}",
        "acc_a":       round(acc_l1, 4),
        "acc_b":       round(acc_l2, 4),
        "n10_a_only":  "—",
        "n01_b_only":  "—",
        "chi2":        "—",
        "p":           "—",
    })

mcnemar_df = pd.DataFrame(mcnemar_rows)
mcnemar_df.to_csv("results/tables/classifier_mcnemar.csv", index=False)
print("✓ McNemar results →", "results/tables/classifier_mcnemar.csv")
print(mcnemar_df[["comparison", "acc_a", "acc_b", "p"]].to_string(index=False))