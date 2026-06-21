"""CardioformerCKD: multimodal prognostic model for incident CKD.

Architecture
------------
        ECG window (B, L, 12) ──► Cardioformer backbone ──► z_ecg (B, G*d_model)
                                                               │
   EHR features (static / seq) ──► EHR encoder ──► z_ehr (B, d_ehr)
                                                               │
                          ┌────────── fusion (concat | gated | cross-attn) ──────────┐
                          ▼                                                            ▼
                   fused (B, d_fuse)                                        prognostic head(s)
                                                                  binary horizon  /  discrete-time survival

The model is *prognostic*: features come from a window strictly before the index
time; labels describe CKD onset within a future horizon. Leakage control lives in
the cohort builder (``data_preprocessing/build_cohort.py``), not here.

``configs`` reuses the Cardioformer attributes (see models/Cardioformer.py) plus:
ehr_in_dim, ehr_mode ('mlp'|'seq'), fusion ('concat'|'gated'|'cross'),
head ('binary'|'survival'), n_intervals (survival only), use_ecg, use_ehr.
"""

import torch
import torch.nn as nn

from models.Cardioformer import Model as CardioformerBackbone
from models.ehr_encoder import EHRMLPEncoder, EHRSequenceEncoder
from models.heads import BinaryHorizonHead, DiscreteTimeSurvivalHead


class GatedFusion(nn.Module):
    """Modality gating: learn per-dimension weights to combine the two embeddings."""

    def __init__(self, d_ecg, d_ehr, d_out):
        super().__init__()
        self.proj_ecg = nn.Linear(d_ecg, d_out)
        self.proj_ehr = nn.Linear(d_ehr, d_out)
        self.gate = nn.Linear(d_out * 2, d_out)

    def forward(self, z_ecg, z_ehr):
        a = self.proj_ecg(z_ecg)
        b = self.proj_ehr(z_ehr)
        g = torch.sigmoid(self.gate(torch.cat([a, b], dim=-1)))
        return g * a + (1 - g) * b


class CrossAttentionFusion(nn.Module):
    """EHR embedding attends over per-granularity ECG tokens (and vice versa)."""

    def __init__(self, d_ecg_token, d_ehr, d_out, n_heads=4):
        super().__init__()
        self.q = nn.Linear(d_ehr, d_out)
        self.kv = nn.Linear(d_ecg_token, d_out)
        self.attn = nn.MultiheadAttention(d_out, n_heads, batch_first=True)
        self.out = nn.Linear(d_out + d_ehr, d_out)

    def forward(self, ecg_tokens, z_ehr):
        # ecg_tokens: (B, G, d_ecg_token); z_ehr: (B, d_ehr)
        q = self.q(z_ehr).unsqueeze(1)          # (B,1,d_out)
        kv = self.kv(ecg_tokens)                # (B,G,d_out)
        ctx, _ = self.attn(q, kv, kv)           # (B,1,d_out)
        ctx = ctx.squeeze(1)
        return self.out(torch.cat([ctx, z_ehr], dim=-1))


class CardioformerCKD(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.use_ecg = getattr(configs, "use_ecg", True)
        self.use_ehr = getattr(configs, "use_ehr", True)
        self.fusion_mode = getattr(configs, "fusion", "gated")
        self.head_type = getattr(configs, "head", "survival")
        self.d_model = configs.d_model

        d_fuse = getattr(configs, "d_fuse", 256)

        # --- ECG branch -----------------------------------------------------
        if self.use_ecg:
            backbone_cfg = configs
            setattr(backbone_cfg, "task_name", "representation")
            self.ecg = CardioformerBackbone(backbone_cfg)
            self.n_gran = self.ecg.n_gran
            d_ecg = self.n_gran * configs.d_model
        else:
            self.ecg = None
            d_ecg = 0

        # --- EHR branch -----------------------------------------------------
        if self.use_ehr:
            if getattr(configs, "ehr_mode", "mlp") == "seq":
                self.ehr = EHRSequenceEncoder(
                    n_features=configs.ehr_in_dim, d_model=configs.d_model,
                    dropout=getattr(configs, "ehr_dropout", 0.2),
                )
            else:
                self.ehr = EHRMLPEncoder(
                    in_dim=configs.ehr_in_dim, embed_dim=configs.d_model,
                    dropout=getattr(configs, "ehr_dropout", 0.2),
                )
            d_ehr = self.ehr.embed_dim
        else:
            self.ehr = None
            d_ehr = 0

        # --- fusion ---------------------------------------------------------
        if self.use_ecg and self.use_ehr:
            if self.fusion_mode == "concat":
                self.fuse = None
                d_fuse = d_ecg + d_ehr
            elif self.fusion_mode == "gated":
                self.fuse = GatedFusion(d_ecg, d_ehr, d_fuse)
            elif self.fusion_mode == "cross":
                self.fuse = CrossAttentionFusion(configs.d_model, d_ehr, d_fuse)
            else:
                raise ValueError(f"unknown fusion: {self.fusion_mode}")
        else:
            self.fuse = None
            d_fuse = d_ecg + d_ehr

        # --- prognostic head ------------------------------------------------
        if self.head_type == "binary":
            self.head = BinaryHorizonHead(d_fuse)
        elif self.head_type == "survival":
            self.head = DiscreteTimeSurvivalHead(
                d_fuse, n_intervals=getattr(configs, "n_intervals", 8)
            )
        else:
            raise ValueError(f"unknown head: {self.head_type}")

    def _encode_ecg(self, x_ecg):
        feats = self.ecg.encode(x_ecg)                         # (B, G*d_model)
        tokens = feats.view(feats.size(0), self.n_gran, self.d_model)  # for cross-attn
        return feats, tokens

    def forward(self, x_ecg=None, ehr=None, ehr_mask=None):
        z_ecg = z_ehr = None
        ecg_tokens = None

        if self.use_ecg:
            z_ecg, ecg_tokens = self._encode_ecg(x_ecg)
        if self.use_ehr:
            if isinstance(self.ehr, EHRSequenceEncoder):
                z_ehr = self.ehr(ehr, key_padding_mask=ehr_mask)
            else:
                z_ehr = self.ehr(ehr)

        if self.use_ecg and self.use_ehr:
            if self.fusion_mode == "concat":
                fused = torch.cat([z_ecg, z_ehr], dim=-1)
            elif self.fusion_mode == "gated":
                fused = self.fuse(z_ecg, z_ehr)
            else:  # cross
                fused = self.fuse(ecg_tokens, z_ehr)
        elif self.use_ecg:
            fused = z_ecg
        else:
            fused = z_ehr

        return self.head(fused)

    # convenience for survival inference
    def predict_cif(self, x_ecg=None, ehr=None, ehr_mask=None):
        assert self.head_type == "survival"
        logits = self.forward(x_ecg, ehr, ehr_mask)
        surv, cif = DiscreteTimeSurvivalHead.survival_from_hazards(logits)
        return surv, cif
