# extract_text_embeddings.py
import os
import torch
import torchaudio
import argparse
from tqdm import tqdm
from whisper import load_model


def load_whisper():
    # ---- 最小修改：强制在 CPU 加载 ----
    model = load_model("tiny", device="cpu")
    model.eval()
    return model


def extract_text_from_path(path):
    txt_path = path.replace(".wav", ".txt")
    if os.path.exists(txt_path):
        return open(txt_path).read().strip()
    return ""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--wav_root", required=True)
    parser.add_argument("--save_path", required=True)
    args = parser.parse_args()

    whisper_model = load_whisper()

    out = {}

    wavs = []
    for root, dirs, files in os.walk(args.wav_root):
        for f in files:
            if f.endswith(".wav"):
                wavs.append(os.path.join(root, f))

    print(f"[INFO] Found {len(wavs)} wav files")

    for wav_path in tqdm(wavs):

        text = extract_text_from_path(wav_path)
        if len(text.strip()) == 0:
            emb = torch.zeros(512)
            out[wav_path] = emb
            continue

        wav, sr = torchaudio.load(wav_path)
        wav = torchaudio.functional.resample(wav, sr, 16000)
        wav = wav.squeeze(0)

        # ---- 最小修改：确保 wav 在 CPU 上 ----
        wav = wav.cpu()

        with torch.no_grad():
            encoded = whisper_model.encoder(wav.unsqueeze(0))
            # encoded: [1, T, 512]

        emb = encoded.mean(dim=1).squeeze(0)  # [512]

        out[wav_path] = emb

    torch.save(out, args.save_path)
    print(f"[OK] Saved text embeddings → {args.save_path}")


if __name__ == "__main__":
    main()
