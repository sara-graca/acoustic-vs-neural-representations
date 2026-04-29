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

os.makedirs("results/tables", exist_ok=True)
os.makedirs("results/figures", exist_ok=True)


# 1. Parameters & data


with open("params.yaml", encoding="utf-8") as f:
    params = yaml.safe_load(f)

VOWELS = params["tests_inter_distances"]["vowels"]
N_NEURAL_PCS = params["lme_models"]["n_neural_pcs"]   # use first 5 PCs of 50D clustering array

VOWEL_HEIGHT = {
    "i": "high", "y": "high", "u": "high",
    "e": "mid",  "ø": "mid",  "o": "mid", "ə": "mid",
    "ɛ": "low_mid", "ɑ": "low_mid",
    "a": "low",
}

MODELS_NEURAL = {
    "whisper_layer4": "data/features/features_whisper_layer4_pca.npz",
    "xlsr_layer3":    "data/features/features_xlsr_layer3_pca.npz",
}

# Load acoustic data
df = pd.read_csv("data/features/features_acoustic_norm.csv")
df_v = (
    df[df["phoneme"].isin(VOWELS)]
    .dropna(subset=["F1_norm", "F2_norm"])
    .copy()
)
original_index = df_v.index.values
df_v = df_v.reset_index(drop=True)

# Encode predictors
df_v["L2"]   = (df_v["l1_status"] == "ru").astype(int)   # 1 = L2, 0 = L1
df_v["Male"] = (df_v["gender"] == "m").astype(int)        # 1 = male, 0 = female
df_v["vowel_height"] = df_v["phoneme"].map(VOWEL_HEIGHT)

# Drop rows with missing height encoding (shouldn't happen)
df_v = df_v.dropna(subset=["vowel_height"]).reset_index(drop=True)


# 2. Load neural PCs (first N_NEURAL_PCS of 50D)


def load_neural_pcs(npz_path: str, n_pcs: int) -> np.ndarray:
    data  = np.load(npz_path)
    X_all = data["clustering"]              # (len(df), 50)
    X_v   = X_all[original_index]           # vowel rows
    return X_v[:, :n_pcs]                   # first n_pcs components

neural_pcs = {}
for model_name, npz_path in MODELS_NEURAL.items():
    pcs = load_neural_pcs(npz_path, N_NEURAL_PCS)
    for i in range(N_NEURAL_PCS):
        col = f"PC{i+1}_{model_name}"
        df_v[col] = pcs[:, i]


# 3. Model-fitting helpers


def fit_model_sequence(df_fit: pd.DataFrame, response: str) -> dict:
    
    # Standardise response and continuous predictors for numerical stability
    df_scaled = df_fit.copy()
    for col in [response]:
        std = df_scaled[col].std()
        if std > 0:
            df_scaled[col] = (df_scaled[col] - df_scaled[col].mean()) / std

    def fit(formula, re_formula=None):
        model = smf.mixedlm(
            formula,
            data=df_scaled,
            groups=df_scaled["speaker_id"],
            re_formula=re_formula,
        )
        return model.fit(reml=False, method="powell")  # powell more stable than lbfgs

    m0 = fit(f"{response} ~ 1")
    m1 = fit(f"{response} ~ L2 + Male")
    m2 = fit(f"{response} ~ L2 + Male + L2:Male")
    m3 = fit(f"{response} ~ L2 + Male + L2:Male + C(vowel_height)")

    try:
        m4 = fit(f"{response} ~ L2 + Male + L2:Male + C(vowel_height)",
                 re_formula="~L2")
    except Exception:
        m4 = None

    return {"null": m0, "main": m1, "full": m2, "extended": m3, "random_slopes": m4}


def icc(model) -> float:
    """ICC = σ²_u / (σ²_u + σ²_e) from null model."""
    var_u = float(model.cov_re.iloc[0, 0])
    var_e = float(model.scale)
    return round(var_u / (var_u + var_e), 4)


def r2_nakagawa(model, df_fit: pd.DataFrame, response: str) -> tuple[float, float]:
    var_rand = float(model.cov_re.iloc[0, 0])
    var_res  = float(model.scale)
    # Since response is standardised before fitting, var_total ≈ 1
    var_total = 1.0
    var_fix   = max(0.0, var_total - var_rand - var_res)
    total     = var_fix + var_rand + var_res
    if total == 0:
        return 0.0, 0.0
    r2_m = round(var_fix / total, 4)
    r2_c = round((var_fix + var_rand) / total, 4)
    return r2_m, r2_c


def lrt(m_null, m_alt) -> tuple[float, float, int]:
    if m_null is None or m_alt is None:
        return np.nan, np.nan, 0
    lr_stat = 2 * (m_alt.llf - m_null.llf)
    if not np.isfinite(lr_stat):
        return np.nan, np.nan, 0
    df_diff = max(int(m_alt.df_modelwc - m_null.df_modelwc), 1)
    p_val   = float(chi2_dist.sf(lr_stat, df_diff))
    return round(float(lr_stat), 4), round(p_val, 4), df_diff



# 4. Run models for all responses


responses = ["F1_norm"] + [
    f"PC{i+1}_{m}"
    for m in MODELS_NEURAL
    for i in range(N_NEURAL_PCS)
]

all_model_fits  = {}   # response → model dict
summary_rows    = []
lrt_rows        = []

print(f"Fitting LME models for {len(responses)} response variables...")

for response in tqdm(responses):
    if response not in df_v.columns:
        continue

    df_fit = df_v[["speaker_id", "L2", "Male", "vowel_height", response]].dropna()

    fits = fit_model_sequence(df_fit, response)
    all_model_fits[response] = fits

    # ICC from null model
    icc_val = icc(fits["null"])

    # AIC / BIC
    for name, m in fits.items():
        if m is None:
            continue
        summary_rows.append({
            "response":  response,
            "model":     name,
            "AIC":       round(m.aic, 2),
            "BIC":       round(m.bic, 2),
            "logLik":    round(m.llf, 2),
            "ICC":       icc_val if name == "null" else "",
        })

    # LRT between nested pairs
    pairs = [
        ("null",     "main",          "add L2+Male"),
        ("main",     "full",          "add L2:Male"),
        ("full",     "extended",      "add vowel_height"),
        ("extended", "random_slopes", "add random slope L2"),
    ]
    for m_null_name, m_alt_name, label in pairs:
        m0 = fits.get(m_null_name)
        m1 = fits.get(m_alt_name)
        if m0 is None or m1 is None:
            continue
        lr, p, df_d = lrt(m0, m1)
        lrt_rows.append({
            "response":   response,
            "comparison": label,
            "LR":         lr,
            "df":         df_d,
            "p":          p,
        })

    # R² for extended model
    r2_m, r2_c = r2_nakagawa(fits["extended"], df_fit, response)
    summary_rows_r2 = [r for r in summary_rows
                       if r["response"] == response and r["model"] == "extended"]
    if summary_rows_r2:
        summary_rows_r2[-1]["R2_marginal"]    = r2_m
        summary_rows_r2[-1]["R2_conditional"] = r2_c

summary_df = pd.DataFrame(summary_rows)
lrt_df     = pd.DataFrame(lrt_rows)

# Best model per response based on AIC
print("\nBest model per response (lowest AIC):")
best_rows = []
for response in responses:
    sub = summary_df[summary_df["response"] == response]
    if sub.empty:
        continue
    best = sub.loc[sub["AIC"].idxmin()]
    best_rows.append({
        "response": response,
        "best_model": best["model"],
        "AIC": best["AIC"],
    })
best_df = pd.DataFrame(best_rows)
best_df.to_csv("results/tables/lme_best_model.csv", index=False)
print(best_df.to_string(index=False))

summary_df.to_csv("results/tables/lme_summary.csv", index=False)
lrt_df.to_csv("results/tables/lme_lrt.csv", index=False)
print("✓ LME summary  →", "results/tables/lme_summary.csv")
print("✓ LRT results  →", "results/tables/lme_lrt.csv")


# 5. Fixed effects table for extended model


fe_rows = []
for response in responses:
    fits = all_model_fits.get(response)
    if fits is None:
        continue
    m = fits["extended"]
    for param, coef, se, pval in zip(
        m.params.index, m.params.values,
        m.bse.values, m.pvalues.values
    ):
        fe_rows.append({
            "response": response,
            "parameter": param,
            "coef": round(float(coef), 4),
            "se":   round(float(se), 4),
            "p":    round(float(pval), 4),
        })

fe_df = pd.DataFrame(fe_rows)
fe_df.to_csv("results/tables/lme_fixed_effects.csv", index=False)
print("✓ Fixed effects →", "results/tables/lme_fixed_effects.csv")


# 6. Forest plot — L2 coefficient across responses


l2_rows = fe_df[fe_df["parameter"] == "L2"].copy()

if not l2_rows.empty:
    from scipy.stats import norm as scipy_norm
    l2_rows["ci_low"]  = l2_rows["coef"] - 1.96 * l2_rows["se"]
    l2_rows["ci_high"] = l2_rows["coef"] + 1.96 * l2_rows["se"]

    fig, ax = plt.subplots(figsize=(7, max(4, len(l2_rows) * 0.4)))
    y_pos = np.arange(len(l2_rows))

    colors = ["steelblue" if "F1" in r else
              "darkorange" if "whisper" in r else
              "seagreen"
              for r in l2_rows["response"]]

    for y, (_, row), c in zip(y_pos, l2_rows.iterrows(), colors):
        ax.plot(row["coef"], y, "o", color=c, zorder=3)
        ax.hlines(y, row["ci_low"], row["ci_high"],
                  color=c, linewidth=2, zorder=2)

    ax.axvline(0, color="grey", linestyle="--", linewidth=0.8)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(l2_rows["response"], fontsize=8)
    ax.set_xlabel("L2 coefficient (95% CI)")
    ax.set_title("Effect of L2 status on acoustic and neural responses\n"
                 "(extended model, ML estimation)")

    patches = [
        mpatches.Patch(color="steelblue",  label="Acoustic F1"),
        mpatches.Patch(color="darkorange", label="Whisper layer 4"),
        mpatches.Patch(color="seagreen",   label="XLS-R layer 3"),
    ]
    ax.legend(handles=patches, fontsize=8)
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    plt.savefig("results/figures/lme_l2_forest.png",
                dpi=150, bbox_inches="tight")
    plt.close()
    print("✓ Forest plot   →", "results/figures/lme_l2_forest.png")


# 7. ICC summary table


icc_rows = []
for response in responses:
    fits = all_model_fits.get(response)
    if fits is None:
        continue
    icc_rows.append({
        "response": response,
        "ICC":      icc(fits["null"]),
    })

icc_df = pd.DataFrame(icc_rows)
icc_df.to_csv("results/tables/lme_icc.csv", index=False)
print("✓ ICC table     →", "results/tables/lme_icc.csv")
print("\nICC values:")
print(icc_df.to_string(index=False))
print("\nLRT results (p < 0.05 = significant improvement):")
print(lrt_df[lrt_df["response"] == "F1_norm"].to_string(index=False))

df_a = df_v[df_v["phoneme"] == "a"][["speaker_id", "F1_norm", "PC1_whisper_layer4"]].dropna()

for response in ["F1_norm", "PC1_whisper_layer4"]:
    m = smf.mixedlm(f"{response} ~ 1", data=df_a, groups=df_a["speaker_id"]).fit(reml=False, method="powell")
    print(response, icc(m))