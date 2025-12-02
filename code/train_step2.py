# /root/taste_assignment/train_step2.py
import torch
from torch.utils.data import DataLoader
import random
from tqdm import tqdm
import matplotlib.pyplot as plt
import pandas as pd
import os

from models import CosyVoiceS3Model, MiniLLM
from dataset_step2 import load_samples, S3Dataset, collate_fn


# =============================
#   Config
# =============================
UTT2_S3_PATH = "/root/autodl-tmp/taste_features/train_s3.pt"
UTT2_TEXT_EMB_PATH = "/root/autodl-tmp/taste_features/train_text_emb.pt"
UTT2_WHISPER_PATH = "/root/autodl-tmp/taste_features/train_whisper_feat.pt"

S3_PAD_ID = 0
S3_VOCAB_SIZE = 4096
LR = 1e-4
WEIGHT_DECAY = 1e-3
NUM_EPOCHS = 5
BATCH_SIZE = 4
TRAIN_RATIO = 0.95

OUT_DIR = "/root/taste_assignment/output_step2"
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(f"{OUT_DIR}/checkpoints", exist_ok=True)


# =============================
#  Training Step
# =============================
def train_one_epoch(model, loader, optimizer, device, epoch, step_log):
    model.train()
    total_loss = 0
    total_tokens = 0

    pbar = tqdm(loader, desc=f"[Train] Epoch {epoch}")

    step_index = 0
    for batch in pbar:
        batch = {k: v.to(device) for k, v in batch.items()}

        loss, logits, _ = model(
            text_emb=batch["text_emb"],
            speech_last=batch["speech_last"],
            speech_mid=batch["speech_mid"],
            speech_mask=batch["speech_mask"],
            text_mask=batch["text_mask"],
            s3_targets=batch["s3_targets"],
        )

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        ntokens = (batch["s3_targets"] != S3_PAD_ID).sum().item()
        total_loss += loss.item() * ntokens
        total_tokens += ntokens

        # ===== record step-level loss + predicted token =====
        pred = logits.argmax(dim=-1)[0].cpu().tolist()
        step_log.append({
            "epoch": epoch,
            "step": step_index,
            "loss": loss.item(),
            "pred_s3_token": pred,
        })

        step_index += 1
        pbar.set_postfix({"loss": f"{loss.item():.4f}"})

    return total_loss / max(1, total_tokens)


# =============================
#  Validation
# =============================
@torch.no_grad()
def eval_one_epoch(model, loader, device, epoch):
    model.eval()
    total_loss = 0
    total_tokens = 0

    pbar = tqdm(loader, desc=f"[Valid] Epoch {epoch}")

    for batch in pbar:
        batch = {k: v.to(device) for k, v in batch.items()}

        loss, _, _ = model(
            text_emb=batch["text_emb"],
            speech_last=batch["speech_last"],
            speech_mid=batch["speech_mid"],
            speech_mask=batch["speech_mask"],
            text_mask=batch["text_mask"],
            s3_targets=batch["s3_targets"],
        )

        ntokens = (batch["s3_targets"] != S3_PAD_ID).sum().item()
        total_loss += loss.item() * ntokens
        total_tokens += ntokens

        pbar.set_postfix({"val_loss": f"{loss.item():.4f}"})

    return total_loss / max(1, total_tokens)


# =============================
#  Test Clean Accuracy
# =============================
@torch.no_grad()
def compute_top1_accuracy(model, test_samples, device):
    from tqdm import tqdm

    model.eval()

    total = 0
    correct = 0

    for s in tqdm(test_samples[:100], desc="[Test-clean] top-1 accuracy"):
        # -------------- 1. 跳过任何不完整样本 --------------
        if "text_emb" not in s or s["text_emb"] is None:
            continue
        if "speech_mid" not in s or s["speech_mid"] is None:
            continue
        if "speech_last" not in s or s["speech_last"] is None:
            continue

        # -------------- 2. 取 S3 token 序列（支持所有命名 + 自动修 shape） --------------
        if "s3_targets" in s:
            t = s["s3_targets"]
        elif "s3_tokens" in s:      # test-clean 样本里是这个键
            t = s["s3_tokens"]
        elif "s3" in s:
            t = s["s3"]
        else:
            continue

        if t is None or t.numel() == 0:
            continue

        # 有的样本是 (1, L)，有的是 (L,)，甚至可能 (1,1,L)
        # 统一 squeeze 到一维 (L,)
        while t.dim() > 1:
            t = t.squeeze(0)

        if t.numel() == 0:
            continue

        # -------------- 3. 取首 token 作为目标 --------------
        target = t[0].item()

        # -------------- 4. 模型前向 --------------
        batch = {
            "text_emb": s["text_emb"].unsqueeze(0).to(device),
            "speech_mid": s["speech_mid"].unsqueeze(0).to(device),
            "speech_last": s["speech_last"].unsqueeze(0).to(device),
            "speech_mask": torch.ones(1, s["speech_mid"].size(0), dtype=torch.bool).to(device),
            "text_mask": torch.ones(1, s["text_emb"].size(0), dtype=torch.bool).to(device),
            # 这里保证传进去的是 [1, L] 的 int tensor
            "s3_targets": t.unsqueeze(0).to(device),
        }

        _, logits, _ = model(**batch)

        # -------------- 5. 取 prefix 后第一个 token 的预测 --------------
        text_len = batch["text_emb"].size(1)
        first_token_idx = 2 + text_len  # [SOS] + text + [TASK]

        pred = logits[0, first_token_idx].argmax(-1).item()

        # -------------- 6. 累计 --------------
        total += 1
        if pred == target:
            correct += 1

    if total == 0:
        print("No valid samples found for accuracy!")
        return 0.0

    acc = correct / total
    print(f"[TEST] top-1 accuracy = {acc:.4f} ({correct}/{total})")
    return acc


# =============================
#              MAIN
# =============================
def main():
    device = torch.device("cpu")  # 按要求强制 CPU
    print(">>> Running Step2 on CPU")

    # ---- load data ----
    samples = load_samples(UTT2_S3_PATH, UTT2_TEXT_EMB_PATH, UTT2_WHISPER_PATH)
    random.shuffle(samples)

    n_train = int(len(samples) * TRAIN_RATIO)
    train_samples = samples[:n_train]
    test_samples = samples[n_train:]

    train_loader = DataLoader(
        S3Dataset(train_samples),
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=lambda b: collate_fn(b, S3_PAD_ID),
    )
    valid_loader = DataLoader(
        S3Dataset(test_samples),
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=lambda b: collate_fn(b, S3_PAD_ID),
    )

    # ---- infer dims ----
    ex = samples[0]
    text_dim = ex["text_emb"].size(-1)
    d_mid = ex["speech_mid"].size(-1)
    d_last = ex["speech_last"].size(-1)

    # Mini LLM
    llm = MiniLLM(hidden_dim=text_dim)

    model = CosyVoiceS3Model(
        llm=llm,
        text_dim=text_dim,
        speech_last_dim=d_last,
        speech_mid_dim=d_mid,
        hidden_dim=text_dim,
        s3_vocab_size=S3_VOCAB_SIZE,
        s3_pad_id=S3_PAD_ID,
        freeze_llm=False,
    ).to(device)

    optim = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=LR,
        weight_decay=WEIGHT_DECAY,
    )

    # logging
    train_losses = []
    valid_losses = []
    step_log = []  # 每个 batch 的 loss + token

    # ----------------------- training ----------------------
    for ep in range(1, NUM_EPOCHS + 1):
        tr = train_one_epoch(model, train_loader, optim, device, ep, step_log)
        va = eval_one_epoch(model, valid_loader, device, ep)

        train_losses.append(tr)
        valid_losses.append(va)

        print(f"Epoch {ep} | train={tr:.4f} | valid={va:.4f}")

        torch.save(model.state_dict(), f"{OUT_DIR}/checkpoints/epoch_{ep}.pt")

    # save step log
    pd.DataFrame(step_log).to_csv(f"{OUT_DIR}/step_log.csv", index=False)
    print(f"Saved batch log → {OUT_DIR}/step_log.csv")

    # save loss curve
    plt.figure(figsize=(7, 5))
    plt.plot(train_losses, label="train")
    plt.plot(valid_losses, label="valid")
    plt.legend()
    plt.title("Step2 Training Loss Curve")
    plt.savefig(f"{OUT_DIR}/loss_curve.png")
    print(f"Saved loss curve → {OUT_DIR}/loss_curve.png")

    # --------------------- test-clean accuracy ---------------------
    acc = compute_top1_accuracy(model, test_samples, device)
    print(f"Top-1 S3 accuracy on test-clean = {acc:.4f}")

    with open(f"{OUT_DIR}/final_accuracy.txt", "w") as f:
        f.write(f"{acc:.4f}")
    print(f"Saved accuracy → {OUT_DIR}/final_accuracy.txt")


if __name__ == "__main__":
    main()
