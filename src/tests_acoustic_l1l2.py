import os
import numpy as np
import pandas as pd
from scipy import stats
import pingouin as pg
from statsmodels.stats.multitest import multipletests
import matplotlib.pyplot as plt
import seaborn as sns

os.makedirs("results/figures", exist_ok=True)
os.makedirs("results/tables", exist_ok=True)

df = pd.read_csv("data/features/features_acoustic_norm.csv")
VOWELS = ["a", "e", "i", "o", "u", "y", "ø", "ɛ", "ɑ", "ə"]
df_v = df[df["phoneme"].isin(VOWELS)].dropna(subset=["F1_norm", "F2_norm"]).copy()

rows = []

for phoneme in VOWELS:
    for formant in ["F1_norm", "F2_norm"]:
        l1 = df_v[(df_v["phoneme"] == phoneme) & (df_v["l1_status"] == "fr")][formant].dropna()
        l2 = df_v[(df_v["phoneme"] == phoneme) & (df_v["l1_status"] == "ru")][formant].dropna()

        if len(l1) < 3 or len(l2) < 3:
            continue

        # normality
        _, p_norm_l1 = stats.shapiro(l1.sample(min(len(l1), 200), random_state=42))
        _, p_norm_l2 = stats.shapiro(l2.sample(min(len(l2), 200), random_state=42))
        normal = p_norm_l1 > 0.05 and p_norm_l2 > 0.05

        # levene
        _, p_levene = stats.levene(l1, l2)
        equal_var = p_levene > 0.05

        # test
        if normal:
            stat, p_val = stats.ttest_ind(l1, l2, equal_var=equal_var)
            test = "t-test"
        else:
            stat, p_val = stats.mannwhitneyu(l1, l2, alternative="two-sided")
            test = "mann-whitney"

        # effect size
        cohens_d = pg.compute_effsize(l1, l2, eftype="cohen")

        rows.append({
            "phoneme":   phoneme,
            "formant":   formant,
            "test":      test,
            "stat":      round(stat, 4),
            "p_raw":     round(p_val, 4),
            "cohens_d":  round(cohens_d, 4),
            "l1_mean":   round(l1.mean(), 4),
            "l2_mean":   round(l2.mean(), 4),
        })

results = pd.DataFrame(rows)

# BH FDR correction
_, p_adj, _, _ = multipletests(results["p_raw"], method="fdr_bh")
results["p_adj"] = p_adj.round(4)
results["significant"] = results["p_adj"] < 0.05

results.to_csv("results/tables/l1_l2_acoustic_tests.csv", index=False)
print(results[["phoneme", "formant", "test", "p_raw", "p_adj", "significant", "cohens_d"]].to_string())

print(df_v["phoneme"].value_counts())