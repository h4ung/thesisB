"""Datasets for the CKD prognosis task.

``MIMICIV_CKD_Dataset`` returns, per cohort row:
    x_ecg : (seq_len, n_leads)  float32   -- a window of the preprocessed ECG
    ehr   : (ehr_dim,)          float32   -- standardised static EHR features
    label : dict with binary + survival targets

ECG handling: the saved .npy is (n_leads, full_len). At train time a random
``seq_len`` window is cropped (and optionally augmented); at eval time the
centre window is used, matching the paper's 1-second segmentation idea while
keeping it configurable.

The EHR StandardScaler and median-imputer are fit on the TRAIN split only and
shared with val/test (passed in via ``scaler``/``imputer``).
"""

import json
import os

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from utils.augmentations import random_augment


class EHRStandardizer:
    """Median impute + z-score, fit on train only."""

    def __init__(self):
        self.median_ = None
        self.mean_ = None
        self.std_ = None
        self.columns_ = None

    def fit(self, X, columns):
        self.columns_ = list(columns)
        X = np.clip(np.asarray(X, dtype=np.float64), -1e6, 1e6)
        self.median_ = np.nan_to_num(np.nanmedian(X, axis=0), nan=0.0)
        Xi = self._impute(X)
        self.mean_ = np.nan_to_num(Xi.mean(axis=0), nan=0.0)        # finite mean
        self.std_  = np.nan_to_num(Xi.std(axis=0),  nan=0.0)        # finite std
        self.std_[self.std_ < 1e-6] = 1.0
        return self

    def _impute(self, X):
        X = X.copy()
        inds = np.where(np.isnan(X))
        X[inds] = np.take(self.median_, inds[1])
        return np.nan_to_num(X, nan=0.0)   

    def transform(self, X):
        std = self.std_.copy()
        std[~np.isfinite(std) | (std < 1e-6)] = 1.0   # catch NaN, inf, AND ~0
        out = (self._impute(X) - self.mean_) / std
        return np.nan_to_num(out, nan=0.0).astype(np.float32)

    def state_dict(self):
        return {k: getattr(self, k).tolist() if isinstance(getattr(self, k), np.ndarray)
                else getattr(self, k) for k in ["median_", "mean_", "std_", "columns_"]}

    def load_state_dict(self, d):
        self.median_ = np.array(d["median_"]); self.mean_ = np.array(d["mean_"])
        self.std_ = np.array(d["std_"]); self.columns_ = d["columns_"]
        return self


class MIMICIV_CKD_Dataset(Dataset):
    def __init__(self, root_path, flag="train", seq_len=250, use_ecg=True,
                 use_ehr=True, scaler=None, ehr_columns=None, augment=True,
                 dataset_file="ckd_dataset_split.parquet"):
        self.root_path = root_path
        self.flag = flag
        self.seq_len = seq_len
        self.use_ecg = use_ecg
        self.use_ehr = use_ehr
        self.augment = augment and flag == "train"

        df = pd.read_parquet(os.path.join(root_path, dataset_file))
        self.df = df[df["split"] == flag].reset_index(drop=True)

        # EHR columns
        col_path = os.path.join(root_path, "ehr_feature_columns.json")
        if ehr_columns is not None:
            self.ehr_columns = ehr_columns
        elif os.path.exists(col_path):
            self.ehr_columns = json.load(open(col_path))
        else:
            self.ehr_columns = [c for c in self.df.columns if c.startswith(("age", "sex", "cmb_"))]

        self.scaler = scaler
        if self.use_ehr and self.scaler is None and flag == "train":
            X = self.df[self.ehr_columns].to_numpy(dtype=float)
            self.scaler = EHRStandardizer().fit(X, self.ehr_columns)

        self.ehr_dim = len(self.ehr_columns)

    # -- ECG window selection --------------------------------------------
    def _get_ecg(self, row):
        arr = np.load(row["ecg_npy"]).astype(np.float32)  # (n_leads, full_len)
        n_leads, full_len = arr.shape
        if full_len >= self.seq_len:
            if self.flag == "train":
                start = np.random.randint(0, full_len - self.seq_len + 1)
            else:
                start = (full_len - self.seq_len) // 2
            arr = arr[:, start:start + self.seq_len]
        else:
            pad = self.seq_len - full_len
            arr = np.pad(arr, ((0, 0), (0, pad)))
        t = torch.from_numpy(arr)
        if self.augment:
            t = random_augment(t)
        return t.transpose(0, 1).contiguous()  # (seq_len, n_leads) for TSLib layout

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        item = {}
        if self.use_ecg:
            item["x_ecg"] = self._get_ecg(row)
        if self.use_ehr:
            X = row[self.ehr_columns].to_numpy(dtype=float)[None, :]
            item["ehr"] = torch.from_numpy(self.scaler.transform(X)[0])

        item["label_binary"] = torch.tensor(float(row["label_binary"]))
        item["eligible_binary"] = torch.tensor(float(row.get("eligible_binary", 1.0)))
        item["event_bin"] = torch.tensor(int(row["event_bin"]))
        item["event_indicator"] = torch.tensor(float(row["event_indicator"]))
        item["event_time_days"] = torch.tensor(float(row["event_time_days"]))
        return item
