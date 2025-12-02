#!/usr/bin/env python3
import os
import torch
import torchaudio
from tqdm import tqdm
from transformers import WhisperProcessor, WhisperModel

import argparse


def load_whisper_model(model_dir):
    processor = WhisperProcessor.from_pretrained(model_dir)
    model = WhisperModel.from_pretrained(model_dir)
    model.eval().to("cpu")
    return processor, model


def extract_feats(processor, model, wav_path):
    wav, sr = torchaudio.load(wav_path)
    if sr != 16000:
        wav = torchaudio.functional.resample(wav, sr, 16000)
    wav = wav.mean(0)  # mono waveform

    # ---- 必须加 decoder_input_ids，否则 huggingface whisper 报错 ----
    inputs = processor(
        wav,
        sampling_rate=16000,
        return_tensors="pt"
    )
    inputs["decoder_input_ids"] = torch.tensor([[1]])   # <|startoftranscript|> token

    with torch.no_grad():
        out = model(**inputs)

    # HuggingFace whisper DOES NOT provide two states; only last_hidden_state.
    hidden = out.last_hidden_state.squeeze(0).cpu()

    # We fake "mid" and "final"
    mid = hidden
    final = hidden.mean(dim=0, keepdim=True)  # [1, 512]

    return mid, final


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", required=True)
    parser.add_argument("--save_path", required=True)
    args = parser.parse_args()

    whisper_dir = "/root/autodl-tmp/models/whisper-large-v3"

    processor, model = load_whisper_model(whisper_dir)

    utt2mid = {}
    utt2final = {}

    all_wavs = []
    for root, dirs, files in os.walk(args.data_root):
        for f in files:
            if f.endswith(".wav"):
                all_wavs.append(os.path.join(root, f))

    print(f"[INFO] Found {len(all_wavs)} wavs in {args.data_root}")

    for wav in tqdm(all_wavs, desc="Extract Whisper"):
        try:
            mid, final = extract_feats(processor, model, wav)
            utt2mid[wav] = mid
            utt2final[wav] = final
        except Exception as e:
            print(f"[WARN] failed {wav}: {e}")
            continue

    torch.save({"mid": utt2mid, "final": utt2final}, args.save_path)
    print(f"[INFO] Saved whisper feats → {args.save_path}")


if __name__ == "__main__":
    main()
