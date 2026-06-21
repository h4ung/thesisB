"""Cardioformer ECG backbone.

Re-implementation of the Cardioformer encoder (Mobin et al., 2025) following the
Time-Series-Library ``Model(configs)`` convention so it can be used as a drop-in
ECG encoder. The original repo exposes a ``classification`` task head; here the
backbone is reusable for both classification and as a feature extractor for the
multimodal prognostic model (``models.CardioformerCKD``).

Expected ``configs`` attributes
-------------------------------
enc_in           : number of ECG leads (channels), e.g. 12
seq_len          : number of timestamps in an ECG window
d_model          : token / model dimension (paper: 128)
n_heads          : attention heads
e_layers         : number of encoder layers (paper: 6)
d_ff             : feed-forward hidden dim (paper: 256)
dropout          : dropout rate
patch_len_list   : comma-separated granularities, e.g. "2,4,8,8,16,16,16,16,32"
num_class        : number of classes (classification task only)
cross_channel    : bool, use cross-channel patching (default True)
"""

import torch
import torch.nn as nn

from layers.Embed import MultiGranularityEmbedding
from layers.Cardioformer_EncDec import CardioformerEncoder


def _parse_patch_lens(configs):
    raw = getattr(configs, "patch_len_list", "2,4,8,8,16,16,16,16,32")
    if isinstance(raw, (list, tuple)):
        return [int(p) for p in raw]
    return [int(p) for p in str(raw).split(",") if str(p).strip()]


class Model(nn.Module):
    """Cardioformer backbone. Set ``task_name='classification'`` for the original
    ECG-classification behaviour, or use ``encode()`` to obtain a pooled embedding
    for downstream multimodal fusion."""

    def __init__(self, configs):
        super().__init__()
        self.task_name = getattr(configs, "task_name", "classification")
        self.d_model = configs.d_model
        self.patch_lens = _parse_patch_lens(configs)
        self.n_gran = len(self.patch_lens)

        self.embedding = MultiGranularityEmbedding(
            n_leads=configs.enc_in,
            patch_lens=self.patch_lens,
            d_model=configs.d_model,
            dropout=configs.dropout,
            resnet_hidden=getattr(configs, "resnet_hidden", 64),
            resnet_blocks=getattr(configs, "resnet_blocks", 2),
            cross_channel=getattr(configs, "cross_channel", True),
        )
        self.encoder = CardioformerEncoder(
            n_layers=configs.e_layers,
            d_model=configs.d_model,
            n_heads=configs.n_heads,
            d_ff=configs.d_ff,
            dropout=configs.dropout,
        )
        self.embed_dim = configs.d_model  # per-granularity pooled dim

        if self.task_name == "classification":
            self.act = nn.GELU()
            self.dropout = nn.Dropout(configs.dropout)
            self.projection = nn.Linear(self.n_gran * configs.d_model, configs.num_class)

    # ---- core feature extractor --------------------------------------------
    def encode(self, x_enc):
        """x_enc: (B, seq_len, enc_in) in TSLib layout. Returns (B, G*d_model)."""
        x = x_enc.permute(0, 2, 1).contiguous()   # -> (B, C, L)
        streams = self.embedding(x)               # list of (B, n_g, d_model)
        enc = self.encoder(streams)               # (B, G, d_model)
        return enc.reshape(enc.size(0), -1)       # (B, G*d_model)

    def classification(self, x_enc, x_mark_enc):
        feats = self.encode(x_enc)
        feats = self.dropout(self.act(feats))
        return self.projection(feats)

    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None, mask=None):
        if self.task_name == "classification":
            return self.classification(x_enc, x_mark_enc)
        # default: return embedding
        return self.encode(x_enc)
