"""Prognostic heads.

A *prognostic* CKD model predicts future onset, not the present state. Two heads
are provided:

* ``BinaryHorizonHead`` — single logit for "develops CKD within H days".
  Simple, but discards patients censored before H.

* ``DiscreteTimeSurvivalHead`` — predicts a conditional hazard for each of K
  discrete time intervals. Handles right-censoring properly and yields a full
  survival curve S(t) and cumulative incidence 1 - S(t). This is the recommended
  head for a genuine prognostic model.

See ``utils.losses`` for the matching loss functions and ``utils.metrics`` for
time-dependent evaluation (C-index, time-dependent AUROC).
"""

import torch
import torch.nn as nn


class BinaryHorizonHead(nn.Module):
    def __init__(self, in_dim, hidden=128, dropout=0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)  # (B,) logit


class DiscreteTimeSurvivalHead(nn.Module):
    """Outputs K hazard logits; h_k = P(event in interval k | survived to k)."""

    def __init__(self, in_dim, n_intervals, hidden=128, dropout=0.2):
        super().__init__()
        self.n_intervals = n_intervals
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, n_intervals),
        )

    def forward(self, x):
        return self.net(x)  # (B, K) hazard logits

    @staticmethod
    def survival_from_hazards(hazard_logits):
        """Return survival S(t_k) = prod_{j<=k} (1 - h_j) and CIF = 1 - S."""
        h = torch.sigmoid(hazard_logits)               # (B, K)
        surv = torch.cumprod(1.0 - h, dim=1)           # (B, K)
        cif = 1.0 - surv
        return surv, cif
