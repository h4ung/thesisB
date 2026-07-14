"""Experiment for prognostic CKD prediction with CardioformerCKD."""

import json
import os
import time

import numpy as np
import torch
import torch.optim as optim

from data_provider.data_factory import data_provider
from data_provider.data_loader import EHRStandardizer
from exp.exp_basic import Exp_Basic
from models.CardioformerCKD import CardioformerCKD
from models.heads import DiscreteTimeSurvivalHead
from utils.losses import focal_bce, discrete_time_nll
from utils.metrics import (
    binary_metrics, brier_score, concordance_index, time_dependent_auroc,
)
from utils.tools import EarlyStopping, adjust_learning_rate, set_seed


class Exp_CKD_Prognosis(Exp_Basic):
    def __init__(self, args):
        set_seed(args.seed)
        # discover ehr_in_dim from the train dataset before building the model
        self.train_data, self.train_loader = data_provider(args, "train")
        self._scaler = self.train_data.scaler
        self._ehr_columns = self.train_data.ehr_columns

        # For eval-only runs, prefer the scaler persisted alongside the checkpoint
        # so preprocessing at test time is byte-identical to training.
        if not args.is_training:
            self._load_saved_scaler(os.path.join(args.checkpoints, args.setting,
                                                 "ehr_scaler.json"))

        args.ehr_in_dim = len(self._ehr_columns)
        super().__init__(args)

    def _load_saved_scaler(self, path):
        if not os.path.exists(path):
            return False
        state = json.load(open(path))
        self._scaler = EHRStandardizer().load_state_dict(state)
        self._ehr_columns = list(state["columns_"])
        print(f"[scaler] loaded fitted EHR scaler from {path}")
        return True

    def _build_model(self):
        return CardioformerCKD(self.args)

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

        path = self.checkpoint_dir()
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
            lr_now = optimizer.param_groups[0]["lr"]
            print(f"Epoch {epoch+1} | lr {lr_now:.2e} | train_loss {np.mean(losses):.4f} | "
                  f"val_score {val_score:.4f} | {val_metrics} | "
                  f"{time.time()-t0:.1f}s")

            early(val_score, self.model, path)
            if early.early_stop:
                print("Early stopping")
                break
            adjust_learning_rate(optimizer, epoch + 2, self.args)  # LR for the *next* epoch

        # restore the best weights before returning / testing
        self.load_checkpoint(os.path.join(path, "checkpoint.pth"))
        return self.model

    @torch.no_grad()
    def validate(self, loader):
        self.model.eval()
        probs, labels, elig = [], [], []
        risks, times, events = [], [], []
        horizon_bin = self.args.n_intervals - 1
        for batch in loader:
            x_ecg, ehr = self._to_device(batch)
            out = self.model(x_ecg=x_ecg, ehr=ehr)
            if self.args.head == "binary":
                probs.append(torch.sigmoid(out).cpu().numpy())
                labels.append(batch["label_binary"].numpy())
                elig.append(batch["eligible_binary"].numpy())
            else:
                _, cif = DiscreteTimeSurvivalHead.survival_from_hazards(out)
                cif = cif.cpu().numpy()
                risks.append(cif[:, horizon_bin])          # risk by horizon
                times.append(batch["event_time_days"].numpy())
                events.append(batch["event_indicator"].numpy())
                labels.append(batch["label_binary"].numpy())
                elig.append(batch["eligible_binary"].numpy())

        labels = np.concatenate(labels)
        elig = np.concatenate(elig).astype(bool)

        if self.args.head == "binary":
            probs = np.concatenate(probs)
            m = binary_metrics(labels[elig], probs[elig])
            m["n_eligible"] = int(elig.sum())
            m["n_total"] = int(elig.size)
            return m.get("auroc", float("nan")), m

        risks = np.concatenate(risks)
        times = np.concatenate(times)
        events = np.concatenate(events)

        # C-index uses everyone (right-censoring is handled by the pair definition).
        cidx = concordance_index(times, risks, events)
        # The fixed-horizon metrics may only use patients whose horizon label is
        # actually known: an event by the horizon, or follow-up reaching it.
        # Passing the eligibility mask here is what stops patients censored early
        # from being silently scored as negatives (which inflates the AUROC).
        td_auroc = time_dependent_auroc(risks, labels, eligible_mask=elig)
        brier = brier_score(labels[elig], risks[elig]) if elig.any() else float("nan")

        metrics = {
            "c_index": round(float(cidx), 4),
            "td_auroc": round(float(td_auroc), 4),
            "brier_at_horizon": round(float(brier), 4),
            "n_eligible": int(elig.sum()),
            "n_total": int(elig.size),
        }
        score = cidx if not np.isnan(cidx) else 0.0
        return score, metrics

    @torch.no_grad()
    def test(self):
        # Always evaluate the saved best weights, never whatever happens to be in
        # memory. Required when --is_training 0: without this the model is random.
        self.load_checkpoint()

        _, test_loader = self._get_data("test")
        score, metrics = self.validate(test_loader)
        metrics["checkpoint"] = self.checkpoint_file()
        metrics["setting"] = self.args.setting

        out_dir = os.path.join(self.args.results, self.args.setting)
        os.makedirs(out_dir, exist_ok=True)
        json.dump(metrics, open(os.path.join(out_dir, "test_metrics.json"), "w"), indent=2)
        print(f"[test] score={score:.4f} metrics={metrics}")
        return metrics
