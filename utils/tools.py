"""Misc training utilities (kept compatible with Time-Series-Library)."""

import os
import random

import numpy as np
import torch


def set_seed(seed=41):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def adjust_learning_rate(optimizer, epoch, args):
    if getattr(args, "lradj", "type1") == "type1":
        lr = args.learning_rate * (0.5 ** ((epoch - 1) // 1))
    elif args.lradj == "cosine":
        lr = args.learning_rate / 2 * (1 + np.cos(np.pi * epoch / args.train_epochs))
    else:
        return
    for pg in optimizer.param_groups:
        pg["lr"] = lr


class EarlyStopping:
    """Stop when the monitored score stops improving. Higher == better."""

    def __init__(self, patience=3, verbose=True, delta=0.0, mode="max"):
        self.patience = patience
        self.verbose = verbose
        self.delta = delta
        self.mode = mode
        self.counter = 0
        self.best = None
        self.early_stop = False

    def __call__(self, score, model, path):
        improved = (
            self.best is None
            or (self.mode == "max" and score > self.best + self.delta)
            or (self.mode == "min" and score < self.best - self.delta)
        )
        if improved:
            self.best = score
            self._save(model, path)
            self.counter = 0
        else:
            self.counter += 1
            if self.verbose:
                print(f"EarlyStopping counter: {self.counter}/{self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True

    def _save(self, model, path):
        os.makedirs(path, exist_ok=True)
        torch.save(model.state_dict(), os.path.join(path, "checkpoint.pth"))
        if self.verbose:
            print(f"  -> validation improved ({self.best:.5f}); checkpoint saved")
