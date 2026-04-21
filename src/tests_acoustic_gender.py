import os
import numpy as np
import pandas as pd
from scipy import stats
import pingouin as pg
from statsmodels.stats.multitest import multipletests

os.makedirs("results/tables", exist_ok=True)

df = pd.read_csv("data/features/features_acoustic_norm.csv")
VOWELS = ["a", "e", "i", "o", "u", "y", "ø", "œ", "ɛ", "ɑ", "ə"]
df_v = df[df["phoneme"].isin(VOWELS)].dropna(subset=["F1_norm", "F2_norm"]).copy()

# compute per-speaker per-phoneme mean
speaker_means = df_v.groupby(["speaker_id", "phoneme", "gender"])[["F1_norm", "F2_norm"]].mean().reset_index()

rows = []
for phoneme in VOWELS:
    for formant in ["F1_norm", "F2_norm"]:
        ph = speaker_means[speaker_means["phoneme"] == phoneme]
        female = ph[ph["gender"] == "f"][formant].dropna()
        male   = ph[ph["gender"] == "m"][formant].dropna()

        if len(female) < 3 or len(male) < 3:
            continue

        _, p_norm_f = stats.shapiro(female)
        _, p_norm_m = stats.shapiro(male)
        normal = p_norm_f > 0.05 and p_norm_m > 0.05

        if normal:
            stat, p_val = stats.ttest_ind(female, male)
            test = "t-test"
        else:
            stat, p_val = stats.mannwhitneyu(female, male, alternative="two-sided")
            test = "mann-whitney"

        cohens_d = pg.compute_effsize(female, male, eftype="cohen")

        rows.append({
            "phoneme":  phoneme,
            "formant":  formant,
            "test":     test,
            "stat":     round(stat, 4),
            "p_raw":    round(p_val, 4),
            "cohens_d": round(cohens_d, 4),
        })

results = pd.DataFrame(rows)
_, p_adj, _, _ = multipletests(results["p_raw"], method="fdr_bh")
results["p_adj"]        = p_adj.round(4)
results["significant"]  = results["p_adj"] < 0.05

results.to_csv("results/tables/gender_acoustic_tests.csv", index=False)
print(results[["phoneme", "formant", "test", "p_raw", "p_adj", "significant", "cohens_d"]].to_string())
