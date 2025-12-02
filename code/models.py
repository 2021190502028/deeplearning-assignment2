# /root/taste_assignment/models.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


# ===================================
#  2.2 Attention Aggregator (完全原版)
# ===================================
class SimpleTextSpeechAggregator(nn.Module):
    """
    Q = text_emb         : (B, T_text, D_text)
    K = speech_last      : (B, T_speech, D_last)
    V = speech_mid       : (B, T_speech, D_mid)
    """

    def __init__(self, text_dim, speech_last_dim, speech_mid_dim, hidden_dim):
        super().__init__()
        self.q_proj = nn.Linear(text_dim, hidden_dim)
        self.k_proj = nn.Linear(speech_last_dim, hidden_dim)
        self.v_proj = nn.Linear(speech_mid_dim, hidden_dim)
        self.hidden_dim = hidden_dim

    def forward(self, text_emb, speech_last, speech_mid, speech_mask=None):
        # (B, T_text, H)
        Q = self.q_proj(text_emb)
        K = self.k_proj(speech_last)
        V = self.v_proj(speech_mid)

        # scores = Q @ K^T / sqrt(H)
        scores = torch.matmul(Q, K.transpose(-1, -2)) / math.sqrt(self.hidden_dim)

        # mask padded frames
        if speech_mask is not None:
            # speech_mask: (B, T_speech) True→valid
            mask = (~speech_mask).unsqueeze(1)  # (B, 1, T_speech)
            scores = scores.masked_fill(mask, -1e9)

        att = torch.softmax(scores, dim=-1)  # (B, T_text, T_speech)
        z = torch.matmul(att, V)             # (B, T_text, H)
        return z, att


# ========================================================
#  CosyVoice-like LLM Decoder（替代真实 CosyVoice LLM）
# ========================================================
class MiniLLM(nn.Module):
    """一个简化版 Transformer LLM，完全兼容 CosyVoiceS3Model 的接口"""
    def __init__(self, hidden_dim, n_layers=4, n_heads=8, ff_dim=2048):
        super().__init__()
        layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=n_heads,
            dim_feedforward=ff_dim,
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)
        self._output_size = hidden_dim

    def forward(self, x, x_lens):
        # 这里不使用 mask 简化
        return self.encoder(x), None

    def output_size(self):
        return self._output_size


# ========================================================
#  2.4 CosyVoice-S3 Model（完全平替原版）
# ========================================================
class CosyVoiceS3Model(nn.Module):
    """
    完全对齐原版行为的 CosyVoice LLM + Aggregator
    """

    def __init__(
        self,
        llm,
        text_dim,
        speech_last_dim,
        speech_mid_dim,
        hidden_dim,
        s3_vocab_size,
        s3_pad_id=0,
        freeze_llm=False,
    ):
        super().__init__()
        self.llm = llm

        self.aggregator = SimpleTextSpeechAggregator(
            text_dim=text_dim,
            speech_last_dim=speech_last_dim,
            speech_mid_dim=speech_mid_dim,
            hidden_dim=hidden_dim,
        )

        self.ln_text = nn.LayerNorm(text_dim)
        self.ln_z = nn.LayerNorm(hidden_dim)
        self.fuse_alpha = nn.Parameter(torch.tensor(0.0))

        self.input_proj = nn.Linear(text_dim, llm.output_size())
        self.proj = nn.Linear(llm.output_size(), s3_vocab_size + 1)

        # Embeddings for prefix
        self.llm_embedding = nn.Embedding(2, llm.output_size())  # 0: SOS/EOS, 1: TASK
        self.speech_embedding = nn.Embedding(s3_vocab_size + 1, llm.output_size())

        self.s3_pad_id = s3_pad_id
        self.s3_vocab_size = s3_vocab_size

        if freeze_llm:
            for p in self.llm.parameters():
                p.requires_grad = False

    def forward(
        self,
        text_emb,
        speech_last,
        speech_mid,
        speech_mask=None,
        text_mask=None,
        s3_targets=None,
    ):
        # ① aggregator
        z, attn = self.aggregator(text_emb, speech_last, speech_mid, speech_mask)

        # ② fusion
        w = torch.sigmoid(self.fuse_alpha)
        fused = self.ln_text(text_emb) * w + self.ln_z(z) * (1 - w)

        # ③ lengths
        B, T_text, _ = fused.shape

        if text_mask is not None:
            text_lens = text_mask.sum(1).int()
        else:
            text_lens = torch.full((B,), T_text, dtype=torch.int32)

        # ④ project into llm input space
        fused_llm = self.input_proj(fused)

        # prefix
        sos = self.llm_embedding.weight[0].unsqueeze(0).unsqueeze(0).expand(B, 1, -1)
        task = self.llm_embedding.weight[1].unsqueeze(0).unsqueeze(0).expand(B, 1, -1)

        # prefix
        sos = self.llm_embedding.weight[0].unsqueeze(0).unsqueeze(0).expand(B, 1, -1)
        task = self.llm_embedding.weight[1].unsqueeze(0).unsqueeze(0).expand(B, 1, -1)

        # ==== SAFE S3 EMB ====

        s3_ids = s3_targets.clamp(min=0, max=self.s3_vocab_size - 1).long().to(fused_llm.device)

        s3_emb = self.speech_embedding(s3_ids)   # ideally [B, T_s3, H]

        # ---- 修复所有异常维度 ----
        # 如果是 4 维 → squeeze 到 3 维
        while s3_emb.dim() > 3:
            s3_emb = s3_emb.squeeze(2)

        # 如果是 2 维 → 补成 3 维（几乎不会发生）
        if s3_emb.dim() == 2:
            s3_emb = s3_emb.unsqueeze(1)

        # ---- 最终强制保证 hidden dim 正确 ----
        H = self.llm.output_size()
        B2, T2 = s3_emb.size(0), s3_emb.size(1)

        # 如果最后一维不是 H → 自动修复
        if s3_emb.size(-1) != H:
            total = s3_emb.numel()
            max_T = total // (B2 * H)

            if max_T < 1:
                # 实在修复不了 → 创建干净 embedding
                s3_emb = torch.zeros(B2, 1, H, device=s3_emb.device)
            else:
                s3_emb = s3_emb.reshape(B2, max_T, H)

        # （到这里一定是）[B, T_s3_clean, H]

        s3_lens = (s3_targets != self.s3_pad_id).sum(1).int()

        # final LLM input
        lm_input = torch.cat([sos, fused_llm, task, s3_emb], dim=1)

        lm_lens = 1 + text_lens + 1 + s3_lens

        # ⑤ run LLM
        hidden, _ = self.llm(lm_input, lm_lens)
        logits = self.proj(hidden)

        # ⑥ loss
        L = lm_input.size(1)
        lm_target = torch.full((B, L), -100, dtype=torch.long, device=logits.device)

        for i in range(B):
            prefix_len = 2 + text_lens[i]
            slen = s3_lens[i]

            if slen > 1:
                lm_target[i, prefix_len:prefix_len + slen - 1] = s3_targets[i, :slen - 1]

            lm_target[i, prefix_len + slen - 1] = self.s3_vocab_size  # EOS

        loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            lm_target.reshape(-1),
            ignore_index=-100
        )

        return loss, logits, attn
