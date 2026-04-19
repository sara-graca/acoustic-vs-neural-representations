import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import yaml
import numpy as np
import pandas as pd
import torch
from transformers import WhisperProcessor, WhisperModel
from tqdm import tqdm

with open("params.yaml") as f:
    params = yaml.safe_load(f)

MODEL_NAME = params["whisper"]["model"]
LAYER      = params["whisper"]["layer"]
INPUT      = os.path.join("data", "features", "features_acoustic.csv")
OUTPUT     = os.path.join("data", "features", "features_whisper.npz")

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")
print(f"Loading {MODEL_NAME}...")

processor = WhisperProcessor.from_pretrained(MODEL_NAME)
model     = WhisperModel.from_pretrained(MODEL_NAME, output_hidden_states=True)
model     = model.to(device)
model.eval()

df = pd.read_csv(INPUT)

embeddings = {}
current_wav, current_audio = None, None

for i, row in tqdm(df.iterrows(), total=len(df)):
    if row["wav_path"] != current_wav:
        current_wav = row["wav_path"]
        import soundfile as sf
        audio, sr = sf.read(current_wav)
        if len(audio.shape) > 1:
            audio = audio.mean(axis=1)
        current_audio = (audio, sr)

    audio, sr = current_audio
    onset_sample  = int(row["onset"] * sr)
    offset_sample = int(row["offset"] * sr)
    segment = audio[onset_sample:offset_sample]

    if len(segment) < 10:
        embeddings[i] = np.zeros(model.config.d_model)
        continue

    inputs = processor(segment, sampling_rate=sr, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model.encoder(
            inputs.input_features,
            output_hidden_states=True
        )

    hidden = outputs.hidden_states[LAYER]
    seq_len = hidden.shape[1]

    total_frames = inputs.input_features.shape[-1] // 2
    duration_frames = max(1, int((row["offset"] - row["onset"]) * total_frames / 30.0))
    start_frame = min(int(row["onset"] * total_frames / 30.0), seq_len - 1)
    end_frame   = min(start_frame + duration_frames, seq_len)

    token_hidden = hidden[0, start_frame:end_frame, :]
    embedding    = token_hidden.mean(dim=0).cpu().numpy()
    embeddings[i] = embedding

os.makedirs(os.path.dirname(OUTPUT), exist_ok=True)
np.savez(OUTPUT, **{str(k): v for k, v in embeddings.items()})
print(f"Done: {len(embeddings)} embeddings → {OUTPUT}")