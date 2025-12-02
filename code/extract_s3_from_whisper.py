#!/usr/bin/env python3
import os
import torch
import onnxruntime as ort
from tqdm import tqdm


def load_s3_tokenizer(model_dir):
    onnx_path = os.path.join(model_dir, "speech_tokenizer_v1.onnx")
    sess = ort.InferenceSession(
        onnx_path,
        providers=["CPUExecutionProvider"]
    )
    return sess


def extract_from_whisper(sess, feats):
    """
    feats: (T, 512) torch tensor
    ONNX expects:
        feats: [1, 128, T]
        feats_length: [1]
    """

    # ----------- (1) skip empty tensors -----------
    if feats is None or feats.numel() == 0:
        raise ValueError("empty whisper feature")

    # feats: (T,512)
    feats = feats.transpose(0, 1).unsqueeze(0)  # (1,512,T)

    # ----------- (2) 如果维度不足 128 → 自动 pad -----------
    if feats.size(1) < 128:
        # pad 到 [1,128,T]
        pad = torch.zeros(1, 128 - feats.size(1), feats.size(2))
        feats = torch.cat([feats, pad], dim=1)

    # ----------- (3) 只取前 128 -----------
    feats = feats[:, :128, :]  # (1,128,T)

    L = feats.size(-1)
    feats_length = torch.tensor([L], dtype=torch.int32)

    inputs = {
        sess.get_inputs()[0].name: feats.cpu().numpy(),
        sess.get_inputs()[1].name: feats_length.cpu().numpy(),
    }

    out = sess.run(None, inputs)
    s3 = out[0]   # (1, L)
    return torch.tensor(s3).squeeze(0)   # (L,)


def main():
    cosy_dir = "/root/autodl-tmp/modelscope_cache/iic/CosyVoice-300M"
    whisper_path = "/root/autodl-tmp/taste_features/test_whisper_feat.pt"
    save_path = "/root/autodl-tmp/taste_features/test_s3.pt"

    sess = load_s3_tokenizer(cosy_dir)
    print(f"[INFO] Loaded S3 tokenizer → {cosy_dir}")

    whisper = torch.load(whisper_path, map_location="cpu")
    mid = whisper["mid"]

    results = {}
    failed = 0

    for utt, feats in tqdm(mid.items(), desc="Extract S3 from whisper"):
        try:
            s3 = extract_from_whisper(sess, feats)
            results[utt] = s3.unsqueeze(0)  # match train format
        except Exception as e:
            print(f"[WARN] failed {utt}: {e}")
            results[utt] = None
            failed += 1

    torch.save(results, save_path)
    print(f"[INFO] Saved test S3 → {save_path}  (total={len(results)}, failed={failed})")


if __name__ == "__main__":
    main()
