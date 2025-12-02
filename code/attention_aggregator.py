import torch
import torch.nn as nn
import torch.nn.functional as F

class AttentionAggregator(nn.Module):
    """
    Align speech embeddings (T×D_s) to text embeddings (L×D_t)
    Output: z with shape (L×D_t)
    """

    def __init__(self, dim_text, dim_speech):
        super().__init__()

        # 如果 Whisper 的维度 != text embedding 维度 → 需要适配器
        if dim_text != dim_speech:
            self.adapter = nn.Linear(dim_speech, dim_text)
        else:
            self.adapter = nn.Identity()

        self.attn = nn.MultiheadAttention(
            embed_dim=dim_text,
            num_heads=8,
            batch_first=True
        )

    def forward(self, v, speech_hid):
        """
        v: (L, D_text)
        speech_hid: (T, D_speech)
        """

        speech_hid = self.adapter(speech_hid)  # (T, D_text)

        # 注意：MultiheadAttention expects (B, L, D)
        q = v.unsqueeze(0)
        k = speech_hid.unsqueeze(0)
        v_s = speech_hid.unsqueeze(0)

        out, _ = self.attn(q, k, v_s)  # (1, L, D)
        return out.squeeze(0)
