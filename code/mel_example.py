import torchaudio
import matplotlib.pyplot as plt
import os
import numpy as np

# 1) Choose any wav file from your processed dataset
sample_wav = "/root/autodl-tmp/taste_data/wav_16k/test-clean/1089/134686/1089-134686-0000.wav"

wav, sr = torchaudio.load(sample_wav)
wav = wav.mean(0)  # mono

# 2) Compute mel spectrogram
mel_spec = torchaudio.transforms.MelSpectrogram(
    sample_rate=sr,
    n_fft=1024,
    hop_length=256,
    n_mels=80
)(wav)

mel_spec = torchaudio.functional.amplitude_to_DB(mel_spec, multiplier=10, amin=1e-10, db_multiplier=0)

# 3) Ensure save directory exists
os.makedirs("figs", exist_ok=True)

# 4) Plot
plt.figure(figsize=(10, 4))
plt.imshow(mel_spec.numpy(), aspect='auto', origin='lower', cmap='magma')
plt.colorbar(label="dB")
plt.title("Mel-spectrogram Example")
plt.xlabel("Frames")
plt.ylabel("Mels")
plt.tight_layout()

# 5) Save
plt.savefig("figs/mel_example.png", dpi=200)

print("Saved → figs/mel_example.png")
