# src/parse_corpus.py
import os
import re
import yaml
import tgt
import chardet
import pandas as pd
from collections import defaultdict

with open("params.yaml") as f:
    params = yaml.safe_load(f)

CORPUS_DIR    = os.path.join("data", "raw", "ru-fr_interference", "2", "wav_et_textgrids", "FRcorp_textgrids_only")
METADATA_PATH = os.path.join("data", "raw", "ru-fr_interference", "2", "metadata_RUFR.csv")
OUTPUT_PATH   = os.path.join("data", "parsed", "phonemes.csv")

meta          = pd.read_csv(METADATA_PATH, sep=";")
meta["spk_lower"] = meta["spk"].str.lower()
speaker_info  = meta.set_index("spk_lower")[["L1", "Gender"]].to_dict(orient="index")

def read_txt(path):
    with open(path, "rb") as f:
        raw = f.read()
    encoding = chardet.detect(raw)["encoding"]
    return raw.decode(encoding).strip()

rows = []
repetition_counter = defaultdict(lambda: defaultdict(int))

for speaker_folder in sorted(os.listdir(CORPUS_DIR)):
    speaker_path = os.path.join(CORPUS_DIR, speaker_folder)
    if not os.path.isdir(speaker_path):
        continue

    spk_id = speaker_folder.lower()
    if spk_id not in speaker_info:
        print(f"Warning: no metadata for speaker {spk_id}, skipping")
        continue

    l1     = speaker_info[spk_id]["L1"]
    gender = speaker_info[spk_id]["Gender"]

    for filename in sorted(os.listdir(speaker_path)):
        if not filename.endswith(".TextGrid"):
            continue

        match = re.match(
            r"(?P<spk>[^_]+)_(?P<l1>[^_]+)_(?P<list>[^_]+)_(?P<sent_id>FRcorp\d+)\.TextGrid",
            filename, re.IGNORECASE
        )
        if not match:
            print(f"Warning: could not parse filename {filename}, skipping")
            continue

        tg_path  = os.path.join(speaker_path, filename)
        wav_path = tg_path.replace(".TextGrid", ".wav")
        txt_path = tg_path.replace(".TextGrid", ".txt")

        # read sentence text
        try:
            sentence_text = read_txt(txt_path)
        except Exception:
            sentence_text = match.group("sent_id")

        # assign repetition index
        repetition_counter[spk_id][sentence_text] += 1
        rep_idx = repetition_counter[spk_id][sentence_text]

        try:
            tg = tgt.io.read_textgrid(tg_path)
            phones_tier = tg.get_tier_by_name("phones")
        except Exception as e:
            print(f"Warning: skipping {tg_path}: {e}")
            continue

        for interval in phones_tier.intervals:
            label = interval.text.strip()
            # clean malformed labels like "ding... d" → "d"
            label = re.sub(r"^ding+\.+\s*", "", label)
            if label == "":
                continue

            rows.append({
                "speaker_id":    spk_id,
                "l1_status":     l1,
                "gender":        gender,
                "sentence_text": sentence_text,
                "repetition":    rep_idx,
                "phoneme":       label,
                "onset":         interval.start_time,
                "offset":        interval.end_time,
                "duration_ms":   round((interval.end_time - interval.start_time) * 1000, 4),
                "wav_path":      wav_path,
            })

df = pd.DataFrame(rows)
os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

# build sentence index mapping
sentence_texts = sorted(df["sentence_text"].unique())
sentence_index = {text: i+1 for i, text in enumerate(sentence_texts)}
df["sentence_id"] = df["sentence_text"].map(sentence_index)

df.to_csv(OUTPUT_PATH, index=False)
print(f"Done: {len(df)} phoneme tokens → {OUTPUT_PATH}")
print(f"Speakers: {df['speaker_id'].nunique()}")
print(f"Sentences: {df['sentence_text'].nunique()}")
print(f"Repetitions per speaker per sentence: {df.groupby(['speaker_id', 'sentence_text'])['repetition'].max().unique()}")