import os
import re
import yaml
import tgt
import pandas as pd

# config 
with open("params.yaml") as f:
    params = yaml.safe_load(f)

CORPUS_DIR = os.path.join(
    "data", "raw", "ru-fr_interference", "2",
    "wav_et_textgrids", "FRcorp_textgrids_only"
)
METADATA_PATH = os.path.join(
    "data", "raw", "ru-fr_interference", "2", "metadata_RUFR.csv"
)
OUTPUT_PATH = os.path.join("data", "parsed", "phonemes.csv")

# load metadata
meta = pd.read_csv(METADATA_PATH, sep=";")
# build a lookup dict: speaker_id (lowercase) → {l1, gender}
meta["spk_lower"] = meta["spk"].str.lower()
speaker_info = meta.set_index("spk_lower")[["L1", "Gender"]].to_dict(orient="index")

# parse TextGrids 
rows = []

for speaker_folder in sorted(os.listdir(CORPUS_DIR)):
    speaker_path = os.path.join(CORPUS_DIR, speaker_folder)
    if not os.path.isdir(speaker_path):
        continue

    # speaker ID is the folder name, lowercase
    spk_id = speaker_folder.lower()
    if spk_id not in speaker_info:
        print(f"Warning: no metadata for speaker {spk_id}, skipping")
        continue

    l1 = speaker_info[spk_id]["L1"]        # "ru" or "fr"
    gender = speaker_info[spk_id]["Gender"] # "f" or "m"

    for filename in sorted(os.listdir(speaker_path)):
        if not filename.endswith(".TextGrid"):
            continue

        # parse sentence ID and repetition from filename
        # pattern: {spk}_{l1}_{list}_{sentence_id}.TextGrid
        # e.g. ab_rus_list1_FRcorp1.TextGrid
        match = re.match(
            r"(?P<spk>[^_]+)_(?P<l1>[^_]+)_(?P<list>[^_]+)_(?P<sent_id>FRcorp\d+)\.TextGrid",
            filename, re.IGNORECASE
        )
        if not match:
            print(f"Warning: could not parse filename {filename}, skipping")
            continue

        sent_id = match.group("sent_id")   # e.g. FRcorp1

        tg_path = os.path.join(speaker_path, filename)
        wav_path = tg_path.replace(".TextGrid", ".wav")

        try:
            tg = tgt.io.read_textgrid(tg_path)
        except Exception as e:
            print(f"Warning: could not read {tg_path}: {e}")
            continue

        # get the phones tier
        try:
            phones_tier = tg.get_tier_by_name("phones")
        except Exception:
            print(f"Warning: no 'phones' tier in {tg_path}, skipping")
            continue

        for interval in phones_tier.intervals:
            label = interval.text.strip()
            if label == "":
                continue  # skip silences

            duration_ms = (interval.end_time - interval.start_time) * 1000

            rows.append({
                "speaker_id": spk_id,
                "l1_status":  l1,
                "gender":     gender,
                "sentence_id": sent_id,
                "phoneme":    label,
                "onset":      interval.start_time,
                "offset":     interval.end_time,
                "duration_ms": duration_ms,
                "wav_path":   wav_path,
            })

# save
df = pd.DataFrame(rows)
os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
df.to_csv(OUTPUT_PATH, index=False)
print(f"Done: {len(df)} phoneme tokens → {OUTPUT_PATH}")
print(df.head())
print(df["phoneme"].value_counts().head(20))