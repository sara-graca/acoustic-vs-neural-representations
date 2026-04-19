import os
import pandas as pd
import chardet

txt_data = []
corpus_dir = "data/raw/ru-fr_interference/2/wav_et_textgrids/FRcorp_textgrids_only"

for speaker in os.listdir(corpus_dir):
    speaker_path = os.path.join(corpus_dir, speaker)
    if not os.path.isdir(speaker_path):
        continue
    for filename in os.listdir(speaker_path):
        if not filename.endswith(".txt"):
            continue
        txt_path = os.path.join(speaker_path, filename)
        with open(txt_path, "rb") as f:
            raw = f.read()
        encoding = chardet.detect(raw)["encoding"]
        text = raw.decode(encoding).strip()
        txt_data.append({"speaker": speaker, "filename": filename, "text": text})

df_txt = pd.DataFrame(txt_data)
print(df_txt["text"].value_counts().head(20))

print(df_txt["text"].nunique())
print(len(df_txt) / df_txt["text"].nunique())
print(df_txt.groupby("speaker")["text"].count())

speaker = "AB"
sentence = "Dis j'en chie trois fois"
print(df_txt[(df_txt["speaker"] == speaker) & (df_txt["text"] == sentence)])