# /root/taste_assignment/dataset_step2.py
import torch
import os


def safe_tensor(x, dim):
    """
    If x is None or empty tensor, return dummy (1, dim).
    Otherwise return as-is.
    """
    if x is None:
        return torch.zeros(1, dim)
    if hasattr(x, "numel") and x.numel() == 0:
        
        return torch.zeros(1, dim)
    return x


def load_samples(utt2s3_path, utt2text_path, utt2whisper_path):
    print(f"[INFO] Loading S3 from {utt2s3_path}")
    utt2s3 = torch.load(utt2s3_path, map_location="cpu")

    print(f"[INFO] Loading text embeddings from {utt2text_path}")
    utt2text = torch.load(utt2text_path, map_location="cpu")

    print(f"[INFO] Loading Whisper features from {utt2whisper_path}")
    utt2whisper = torch.load(utt2whisper_path, map_location="cpu")

    mid_dict = utt2whisper["mid"]
    final_dict = utt2whisper["final"]

    all_keys = sorted(set(utt2s3.keys()) & set(utt2text.keys()) &
                      set(mid_dict.keys()) & set(final_dict.keys()))

    samples = []
    for key in all_keys:
        s3 = utt2s3.get(key, None)
        text_emb = utt2text.get(key, None)
        mid = mid_dict.get(key, None)
        last = final_dict.get(key, None)

        # infer dims
        text_dim = text_emb.size(-1) if text_emb is not None else 512
        mid_dim = mid.size(-1) if (mid is not None and hasattr(mid, "size")) else 512
        last_dim = last.size(-1) if (last is not None and hasattr(last, "size")) else 1024

        # fix invalid tensors
        s3 = s3 if s3 is not None else torch.tensor([0])
        text_emb = safe_tensor(text_emb, text_dim)
        mid = safe_tensor(mid, mid_dim)
        last = safe_tensor(last, last_dim)

        samples.append({
            "utt_id": key,
            "text_emb": text_emb,
            "speech_mid": mid,
            "speech_last": last,
            "s3_tokens": s3,
        })

    print(f"[INFO] Loaded {len(samples)} usable samples")
    return samples


class S3Dataset(torch.utils.data.Dataset):
    def __init__(self, samples):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


def collate_fn(batch, pad_id):
    B = len(batch)

    text_lens = [b["text_emb"].size(0) for b in batch]
    speech_lens = [b["speech_mid"].size(0) for b in batch]
    s3_lens = [b["s3_tokens"].numel() for b in batch]

    max_text = max(text_lens)
    max_speech = max(speech_lens)
    max_s3 = max(s3_lens)

    text_dim = batch[0]["text_emb"].size(-1)
    mid_dim = batch[0]["speech_mid"].size(-1)
    last_dim = batch[0]["speech_last"].size(-1)

    text_emb = torch.zeros(B, max_text, text_dim)
    speech_mid = torch.zeros(B, max_speech, mid_dim)
    speech_last = torch.zeros(B, max_speech, last_dim)
    speech_mask = torch.zeros(B, max_speech, dtype=torch.bool)
    text_mask = torch.zeros(B, max_text, dtype=torch.bool)
    s3_targets = torch.full((B, max_s3), pad_id, dtype=torch.long)

    for i, b in enumerate(batch):
        tt = text_lens[i]
        ts = speech_lens[i]
        ts3 = s3_lens[i]

        text_emb[i, :tt] = b["text_emb"]
        speech_mid[i, :ts] = b["speech_mid"]
        speech_last[i, :ts] = b["speech_last"]
        speech_mask[i, :ts] = True
        text_mask[i, :tt] = True

        s3 = b["s3_tokens"]
        if not torch.is_tensor(s3):
            s3 = torch.tensor(s3, dtype=torch.long)

        s3_targets[i, :ts3] = s3[:ts3]

    return {
        "text_emb": text_emb,
        "speech_mid": speech_mid,
        "speech_last": speech_last,
        "speech_mask": speech_mask,
        "text_mask": text_mask,
        "s3_targets": s3_targets,
    }
