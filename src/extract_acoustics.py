import os
import yaml
import numpy as np
import pandas as pd
import parselmouth
from parselmouth.praat import call

with open("params.yaml") as f:
    params = yaml.safe_load(f)

ac = params["acoustics"]
MAX_FORMANT_F = ac["max_formant_female"]
MAX_FORMANT_M = ac["max_formant_male"]
N_FORMANTS    = ac["n_formants"]

INPUT  = os.path.join("data", "parsed", "phonemes.csv")
OUTPUT = os.path.join("data", "features", "features_acoustic.csv")

df = pd.read_csv(INPUT)
results = []
current_wav, current_sound = None, None

for i, row in df.iterrows():
    if i % 500 == 0:
        print(f"{i}/{len(df)}")

    if row["wav_path"] != current_wav:
        current_wav = row["wav_path"]
        try:
            current_sound = parselmouth.Sound(current_wav)
        except Exception:
            current_sound = None

    if current_sound is None:
        results.append({
            **row,
            "F1": np.nan, "F2": np.nan, "F3": np.nan,
            "f0": np.nan, "scg": np.nan,
        })
        continue

    midpoint    = (row["onset"] + row["offset"]) / 2
    max_formant = MAX_FORMANT_F if row["gender"] == "f" else MAX_FORMANT_M

    try:
        segment  = current_sound.extract_part(
            row["onset"], row["offset"], preserve_times=True
        )

        # Formants
        formants = call(segment, "To Formant (burg)",
                        0, N_FORMANTS, max_formant, 0.025, 50)

        # Pitch
        pitch = call(segment, "To Pitch", 0, 75, 600)

        # Spectral centre of gravity (for consonants)
        spectrum = segment.to_spectrum()
        scg_val  = spectrum.get_centre_of_gravity(power=2)

        def get(obj, *args):
            v = call(obj, *args)
            return v if (v and np.isfinite(v)) else np.nan

        f1  = get(formants, "Get value at time", 1, midpoint, "Hertz", "Linear")
        f2  = get(formants, "Get value at time", 2, midpoint, "Hertz", "Linear")
        f3  = get(formants, "Get value at time", 3, midpoint, "Hertz", "Linear")
        f0  = get(pitch,    "Get value at time", midpoint, "Hertz", "Linear")
        scg = scg_val if (scg_val and np.isfinite(scg_val)) else np.nan

    except Exception:
        f1 = f2 = f3 = f0 = scg = np.nan

    results.append({
        **row,
        "F1": f1, "F2": f2, "F3": f3,
        "f0": f0, "scg": scg,
    })

out = pd.DataFrame(results)
os.makedirs(os.path.dirname(OUTPUT), exist_ok=True)
out.to_csv(OUTPUT, index=False)
print(f"Done: {len(out)} tokens")

for col in ["F1", "F2", "F3", "f0", "scg"]:
    n = out[col].isna().sum()
    print(f"  {col}: {n} missing ({100*n/len(out):.1f}%)")