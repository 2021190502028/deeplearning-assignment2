#!/usr/bin/env python3
import argparse
import json
import os

import torch
import torchaudio
from tqdm import tqdm
from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq


def load_jsonl(path):
    """Load a .jsonl file -> list of dicts."""
    items = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            items.append(json.loads(line))
    return items


def load_audio(path, target_sr=16000):
    """Load audio and resample to target_sr. Return mono waveform (T,) and sr."""
    waveform, sr = torchaudio.load(path)
    if sr != target_sr:
        waveform = torchaudio.transforms.Resample(
            orig_freq=sr, new_freq=target_sr
        )(waveform)
    # mono
    if waveform.shape[0] > 1:
        waveform = waveform.mean(0, keepdim=True)
    return waveform.squeeze(0), target_sr


def extract_whisper_encoder_feats(waveform, model, processor, device, max_duration=30.0):
    """
    Use Whisper encoder to extract hidden states.
    Return:
      mid_layer: (T_enc, D)
      final_layer: (T_enc, D)
    If audio longer than max_duration, return (None, None).
    """
    # waveform is 1D at 16k
    num_seconds = waveform.numel() / 16000.0
    if num_seconds > max_duration:
        return None, None

    audio_np = waveform.numpy()
    inputs = processor(
        audio_np,
        sampling_rate=16000,
        return_tensors="pt",
    )
    input_features = inputs.input_features.to(device)

    with torch.no_grad():
        enc_out = model.model.encoder(
            input_features=input_features,
            output_hidden_states=True,
            return_dict=True,
        )

    hidden_states = enc_out.hidden_states  # list: [layer0, layer1, ..., last]
    # pick middle layer & last layer
    mid_idx = len(hidden_states) // 2
    mid_layer = hidden_states[mid_idx].cpu().squeeze(0)   # (T_enc, D)
    final_layer = hidden_states[-1].cpu().squeeze(0)      # (T_enc, D)
    return mid_layer, final_layer


def extract_text_embedding_whisper(text, processor, model, device):
    """
    Use Whisper tokenizer + embedding layer to get text embedding v.
    Returns:
      v: (L, D)  (sequence of token embeddings)
    """
    # tokenize text
    # We only use input_ids, no need for decoder_input_ids here
    tokenized = processor.tokenizer(
        text,
        return_tensors="pt",
        add_special_tokens=True,
    )
    input_ids = tokenized["input_ids"].to(device)  # (1, L)

    # get embedding matrix
    embedding_layer = model.get_input_embeddings()  # nn.Embedding

    with torch.no_grad():
        emb = embedding_layer(input_ids)  # (1, L, D)

    return emb.squeeze(0).cpu()  # (L, D)


def main(args):
    # -------------------------
    # 1) Load Whisper locally
    # -------------------------
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Loading Whisper processor from: {args.whisper_dir}")
    processor = AutoProcessor.from_pretrained(
        args.whisper_dir,
        local_files_only=True,
    )

    print(f"Loading Whisper model from: {args.whisper_dir}")
    whisper_model = AutoModelForSpeechSeq2Seq.from_pretrained(
        args.whisper_dir,
        local_files_only=True,
    ).to(device)
    whisper_model.eval()

    # -------------------------
    # 2) Load JSONL list
    # -------------------------
    data = load_jsonl(args.jsonl)
    print(f"Loaded {len(data)} items from {args.jsonl}")

    # -------------------------
    # 3) Prepare outputs
    # -------------------------
    utt2text_emb = {}        # audio_path -> tensor (L, D)
    utt2whisper_mid = {}     # audio_path -> tensor (T_enc, D) or None
    utt2whisper_final = {}   # audio_path -> tensor (T_enc, D) or None

    os.makedirs(os.path.dirname(args.output_text), exist_ok=True)
    os.makedirs(os.path.dirname(args.output_whisper), exist_ok=True)

    # -------------------------
    # 4) Loop over utterances
    # -------------------------
    for item in tqdm(data):
        audio_path = item["audio_path"]
        text = item["text"]

        # ---- 4.1 Text embedding v (from Whisper) ----
        try:
            text_emb = extract_text_embedding_whisper(
                text=text,
                processor=processor,
                model=whisper_model,
                device=device,
            )
        except Exception as e:
            print(f"[WARN] Failed to extract text embedding for {audio_path}: {e}")
            text_emb = None
        utt2text_emb[audio_path] = text_emb

        # ---- 4.2 Whisper encoder features (mid + final) ----
        try:
            waveform, _ = load_audio(audio_path, target_sr=16000)
            mid_feat, final_feat = extract_whisper_encoder_feats(
                waveform=waveform,
                model=whisper_model,
                processor=processor,
                device=device,
                max_duration=args.max_duration,
            )
        except Exception as e:
            print(f"[WARN] Failed to extract whisper feats for {audio_path}: {e}")
            mid_feat, final_feat = None, None

        utt2whisper_mid[audio_path] = mid_feat
        utt2whisper_final[audio_path] = final_feat

    # -------------------------
    # 5) Save outputs
    # -------------------------
    torch.save(utt2text_emb, args.output_text)
    print(
        f"Saved text embeddings for {len(utt2text_emb)} items to {args.output_text}"
    )

    whisper_output = {
        "mid": utt2whisper_mid,
        "final": utt2whisper_final,
    }
    torch.save(whisper_output, args.output_whisper)
    print(
        f"Saved Whisper features for {len(utt2whisper_mid)} items to {args.output_whisper}"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--jsonl", type=str, required=True,
        help="Input jsonl with audio_path and text"
    )
    parser.add_argument(
        "--whisper_dir", type=str, required=True,
        help="Local directory of whisper-large-v3"
    )
    parser.add_argument(
        "--output_text", type=str, required=True,
        help="Output .pt file for text embeddings"
    )
    parser.add_argument(
        "--output_whisper", type=str, required=True,
        help="Output .pt file for Whisper encoder features"
    )
    parser.add_argument(
        "--max_duration", type=float, default=30.0,
        help="Max audio length (seconds) to process"
    )

    args = parser.parse_args()
    main(args)
