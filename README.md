
# Text-Aligned Speech Tokenization and Cross-Modal Aggregation

**Author:** Jiaxi Zhong
**Course:** MDS5122 / AIR5011 — Assignment 2

---

## 1. Project Overview

This repository implements a simplified version of a TASTE-style text-aligned speech tokenizer using Whisper encoder features and a CosyVoice-style S3 decoder. The objective is to align speech representations with text length through a cross-attention mechanism, and fine-tune a decoder to predict S3 units using ground-truth tokens extracted from CosyVoice.

The pipeline includes:

1. Preparing LibriSpeech (train-clean-100 and test-clean).
2. Resampling and indexing data into JSONL format.
3. Extracting Whisper encoder features (mid-layer and last-layer).
4. Extracting text embeddings.
5. Extracting S3 ground-truth tokens using the CosyVoice 300M tokenizer.
6. Implementing a cross-attention aggregator to align speech to text length.
7. Combining aligned speech embeddings with text embeddings.
8. Training a CosyVoice-style decoder to predict S3 tokens.
9. Recording training loss and evaluating test-clean top-1 accuracy.

---

## 2. Environment Setup

```bash
conda create -n taste python=3.10 -y
conda activate taste
pip install -r requirements.txt
```

The entire assignment is run on CPU due to CUDA architecture mismatch with the available GPU.

---

## 3. File Structure (Actual Files Used)

```
code/
│
├── prepare_librispeech.py          # Build JSONL index + resample to 16kHz
├── mel_example.py                  # (Optional) Visualise mel-spectrogram
│
├── extract_whisper_feat.py         # Extract Whisper mid + last-layer features
├── extract_text_embeddings.py      # Extract text embeddings
├── extract_s3.py                   # Extract S3 tokens from raw wav
├── extract_s3_from_whisper.py      # Extract S3 tokens from Whisper mid features
│
├── utt2text_and_feature.py         # Utility script (not required in final pipeline)
│
├── attention_aggregator.py         # Cross-attention aggregator implementation
├── models.py                       # Mini LLM + aggregation + fusion model
├── dataset_step2.py                # Data loading, collation, and sample assembly
│
├── train_step2.py                  # Main training script
├── plot_batch_loss.py              # Batch-level loss curve visualisation
│
└── s3.sh                           # Shell script for S3 token extraction
```

All scripts listed above were used to complete the assignment and reproduce the final results.

---

## 4. Data Preparation

### 4.1 Download LibriSpeech

Download:

* **train-clean-100.tar.gz**
* **test-clean.tar.gz**

Place them under:

```
/root/autodl-tmp/taste_data/LibriSpeech/
```

Extract the archives.

---

### 4.2 Resampling to 16 kHz + JSONL Construction

Run:

```bash
python code/prepare_librispeech.py \
    --librispeech_root /root/autodl-tmp/taste_data/LibriSpeech \
    --output_root /root/autodl-tmp/taste_data \
    --split train-clean-100
```

```bash
python code/prepare_librispeech.py \
    --librispeech_root /root/autodl-tmp/taste_data/LibriSpeech \
    --output_root /root/autodl-tmp/taste_data \
    --split test-clean
```

This produces:

```
train-clean-100_16k.jsonl
test-clean_16k.jsonl
wav_16k/train-clean-100/*
wav_16k/test-clean/*
```

---

## 5. Feature Extraction

### 5.1 Whisper Speech Features (mid-layer & last-layer)

```bash
python code/extract_whisper_feat.py \
    --data_root /root/autodl-tmp/taste_data/wav_16k/train-clean-100 \
    --save_path /root/autodl-tmp/taste_features/train_whisper_feat.pt
```

```bash
python code/extract_whisper_feat.py \
    --data_root /root/autodl-tmp/taste_data/wav_16k/test-clean \
    --save_path /root/autodl-tmp/taste_features/test_whisper_feat.pt
```

### 5.2 Text Embeddings

```bash
python code/extract_text_embeddings.py \
    --wav_root /root/autodl-tmp/taste_data/wav_16k/train-clean-100 \
    --save_path /root/autodl-tmp/taste_features/train_text_emb.pt
```

```bash
python code/extract_text_embeddings.py \
    --wav_root /root/autodl-tmp/taste_data/wav_16k/test-clean \
    --save_path /root/autodl-tmp/taste_features/test_text_emb.pt
```

### 5.3 S3 Token Extraction

Ensure CosyVoice-300M exists under:

```
/root/autodl-tmp/modelscope_cache/iic/CosyVoice-300M/
```

Run:

```bash
python code/extract_s3.py \
    --data_root /root/autodl-tmp/taste_data/wav_16k/train-clean-100 \
    --save_path /root/autodl-tmp/taste_features/train_s3.pt
```

For test-clean (generated from Whisper mid features):

```bash
python code/extract_s3_from_whisper.py
```

Outputs:

```
train_s3.pt
test_s3.pt
```

---

## 6. Cross-Attention Aggregator

Implemented in **attention_aggregator.py**:

* Queries ( Q ): text embeddings
* Keys ( K ): Whisper last-layer speech features
* Values ( V ): Whisper mid-layer features

Cross-attention:

[
\alpha = \mathrm{softmax}(QK^\top / \sqrt{d})
]

[
z = \alpha V
]

The output (z) has **text-token length**, forming the aligned speech representation.

A learnable linear adapter ensures dimension compatibility.

---

## 7. Model Integration and Training

Training script:

```bash
python code/train_step2.py
```

It performs:

1. Loading text, speech, and S3 features via `dataset_step2.py`.
2. Applying the aggregator.
3. Fusing text and speech representations.
4. Feeding the fused sequence into a MiniLLM decoder (defined in `models.py`).
5. Training for one epoch using AdamW.
6. Recording per-batch loss and per-epoch loss.
7. Computing top-1 S3 token accuracy on test-clean (first token).

Outputs saved under `output_step2/`:

```
loss_curve.png
batch_loss_curve.png
step_log.csv
final_accuracy.txt
```

---

## 8. Visualising Training Loss

### Epoch-level curve

Already produced automatically:

```
output_step2/loss_curve.png
```

### Batch-level curve

Generate manually:

```bash
python code/plot_batch_loss.py
```

This reads `step_log.csv` and outputs a smoothed loss plot:

```
output_step2/batch_loss_curve.png
```

---

## 9. Evaluation Results

* **Top-1 first-token S3 accuracy (test-clean):** `1.000`
* **Training/validation loss:** decreasing and stable after first few hundred batches.
* **Batch-level curve:** confirms rapid convergence and stable learning dynamics.

---

## 10. Acknowledgement

Certain parts of debugging and documentation were assisted using ChatGPT.
All code, implementation details, and final verifications were manually reviewed and confirmed.

---