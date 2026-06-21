"""Experiment for prognostic CKD prediction with CardioformerCKD."""

import json
import os
import time

import numpy as np
import torch
import torch.optim as optim

from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from models.CardioformerCKD import CardioformerCKD
from models.heads import DiscreteTimeSurvivalHead
from utils.losses import focal_bce, discrete_time_nll
from utils.metrics import binary_metrics, concordance_index, time_dependent_auroc
from utils.tools import EarlyStopping, adjust_learning_rate, set_seed


class Exp_CKD_Prognosis(Exp_Basic):
    def __init__(self, args):
        set_seed(args.seed)
        # discover ehr_in_dim from the train dataset before building the model
        self.train_data, self.train_loader = data_provider(args, "train")
        args.ehr_in_dim = self.train_data.ehr_dim
        self._scaler = self.train_data.scaler
        self._ehr_columns = self.train_data.ehr_columns
        super().__init__(args)

    def _build_model(self):
        model = CardioformerCKD(self.args)
        return model

    def _get_data(self, flag):
        return data_provider(self.args, flag,
                             scaler=self._scaler, ehr_columns=self._ehr_columns)

    # ---- batching helpers ----------------------------------------------
    def _to_device(self, batch):
        x_ecg = batch.get("x_ecg")
        ehr = batch.get("ehr")
        if x_ecg is not None:
            x_ecg = x_ecg.float().to(self.device)
        if ehr is not None:
            ehr = ehr.float().to(self.device)
        return x_ecg, ehr

    def _compute_loss(self, out, batch):
        if self.args.head == "binary":
            y = batch["label_binary"].to(self.device)
            elig = batch["eligible_binary"].to(self.device).bool()
            if elig.sum() == 0:
                return out.sum() * 0.0
            return focal_bce(out[elig], y[elig],
                             alpha=self.args.focal_alpha, gamma=self.args.focal_gamma)
        else:  # survival
            event_bin = batch["event_bin"].to(self.device)
            event_ind = batch["event_indicator"].to(self.device)
            return discrete_time_nll(out, event_bin, event_ind)

    # ---- main loops -----------------------------------------------------
    def train(self):
        _, train_loader = self.train_data, self.train_loader
        _, vali_loader = self._get_data("val")

        path = os.path.join(self.args.checkpoints, self.args.setting)
        os.makedirs(path, exist_ok=True)
        # persist the fitted scaler for inference
        if self._scaler is not None:
            json.dump(self._scaler.state_dict(),
                      open(os.path.join(path, "ehr_scaler.json"), "w"))

        optimizer = optim.Adam(self.model.parameters(), lr=self.args.learning_rate,
                               weight_decay=self.args.weight_decay)
        early = EarlyStopping(patience=self.args.patience, mode="max")

        for epoch in range(self.args.train_epochs):
            self.model.train()
            t0 = time.time()
            losses = []
            for batch in train_loader:
                optimizer.zero_grad()
                x_ecg, ehr = self._to_device(batch)
                out = self.model(x_ecg=x_ecg, ehr=ehr)
                loss = self._compute_loss(out, batch)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 4.0)
                optimizer.step()
                losses.append(loss.item())

            val_score, val_metrics = self.validate(vali_loader)
            print(f"Epoch {epoch+1} | train_loss {np.mean(losses):.4f} | "
                  f"val_score {val_score:.4f} | {val_metrics} | "
                  f"{time.time()-t0:.1f}s")

            early(val_score, self.model, path)
            if early.early_stop:
                print("Early stopping")
                break
            adjust_learning_rate(optimizer, epoch + 1, self.args)

        self.model.load_state_dict(torch.load(os.path.join(path, "checkpoint.pth")))
        return self.model

    @torch.no_grad()
    def validate(self, loader):
        self.model.eval()
        probs, labels, elig = [], [], []
        risks, times, events, cif_h = [], [], [], []
        horizon_bin = self.args.n_intervals - 1
        for batch in loader:
            x_ecg, ehr = self._to_device(batch)
            out = self.model(x_ecg=x_ecg, ehr=ehr)
            if self.args.head == "binary":
                p = torch.sigmoid(out).cpu().numpy()
                probs.append(p)
                labels.append(batch["label_binary"].numpy())
                elig.append(batch["eligible_binary"].numpy())
            else:
                surv, cif = DiscreteTimeSurvivalHead.survival_from_hazards(out)
                cif = cif.cpu().numpy()
                cif_h.append(cif[:, horizon_bin])
                risks.append(cif[:, horizon_bin])          # risk by horizon
                times.append(batch["event_time_days"].numpy())
                events.append(batch["event_indicator"].numpy())
                labels.append(batch["label_binary"].numpy())

        if self.args.head == "binary":
            probs = np.concatenate(probs); labels = np.concatenate(labels)
            elig = np.concatenate(elig).astype(bool)
            m = binary_metrics(labels[elig], probs[elig])
            return m.get("auroc", float("nan")), m
        else:
            risks = np.concatenate(risks); times = np.concatenate(times)
            events = np.concatenate(events); labels = np.concatenate(labels)
            cidx = concordance_index(times, risks, events)
            td_auroc = time_dependent_auroc(np.concatenate(cif_h), labels)
            return (cidx if not np.isnan(cidx) else 0.0,
                    {"c_index": round(cidx, 4), "td_auroc": round(td_auroc, 4)})

    @torch.no_grad()
    def test(self):
        _, test_loader = self._get_data("test")
        score, metrics = self.validate(test_loader)
        out_dir = os.path.join(self.args.results, self.args.setting)
        os.makedirs(out_dir, exist_ok=True)
        json.dump(metrics, open(os.path.join(out_dir, "test_metrics.json"), "w"), indent=2)
        print(f"[test] score={score:.4f} metrics={metrics}")
        return metrics
