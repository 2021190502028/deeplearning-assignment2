#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import torchaudio
from tqdm import tqdm


def load_transcripts(split_root: Path):
    """读取所有 *.trans.txt 文件，生成 utt_id -> text 的映射"""
    mapping = {}
    for txt in split_root.rglob("*.trans.txt"):
        with txt.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                utt_id, text = line.split(" ", 1)
                mapping[utt_id] = text
    return mapping


def process_split(split_name: str, librispeech_root: Path,
                  output_root: Path, max_samples=None):

    split_root = librispeech_root / split_name
    if not split_root.exists():
        raise RuntimeError(f"Not found: {split_root}")

    print(f"Processing {split_name} ...")

    transcript_map = load_transcripts(split_root)
    print(f"Loaded {len(transcript_map)} transcripts")

    # 输出目录
    wav16_dir = output_root / "wav_16k"
    wav16_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = output_root / f"{split_name}_16k.jsonl"

    # 读取所有 FLAC（LibriSpeech 默认）
    audio_files = sorted(list(split_root.rglob("*.flac")))

    n = 0
    with jsonl_path.open("w", encoding="utf-8") as fw:

        for audio_path in tqdm(audio_files):

            utt_id = audio_path.stem  # 形如 19-198-0000
            if utt_id not in transcript_map:
                continue

            text = transcript_map[utt_id]

            wav, sr = torchaudio.load(str(audio_path))

            # 多通道合并为单通道
            if wav.size(0) > 1:
                wav = wav.mean(dim=0, keepdim=True)

            # 重采样到 16 kHz
            if sr != 16000:
                wav = torchaudio.transforms.Resample(
                    orig_freq=sr, new_freq=16000)(wav)
                sr = 16000

            # 输出 wav 路径（保留 LibriSpeech 的结构）
            rel = audio_path.relative_to(librispeech_root).with_suffix(".wav")
            outpath = wav16_dir / rel
            outpath.parent.mkdir(parents=True, exist_ok=True)
            torchaudio.save(str(outpath), wav, sr)

            # 写 jsonl
            item = {
                "utt_id": utt_id,
                "audio_path": str(outpath),
                "text": text
            }
            fw.write(json.dumps(item, ensure_ascii=False) + "\n")

            n += 1
            if max_samples and n >= max_samples:
                break

    print(f"Done {split_name}: {n} samples → {jsonl_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--librispeech_root", type=str, required=True,
                        help="Path to LibriSpeech dataset root")
    parser.add_argument("--output_root", type=str, required=True,
                        help="Directory for saving 16k wav and jsonl")
    parser.add_argument("--split", type=str, default="train-clean-100",
                        help="Split name, e.g. train-clean-100")
    parser.add_argument("--max_samples", type=int, default=None,
                        help="Optional: limit number of samples")

    args = parser.parse_args()

    process_split(
        split_name=args.split,
        librispeech_root=Path(args.librispeech_root),
        output_root=Path(args.output_root),
        max_samples=args.max_samples
    )
