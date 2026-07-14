#!/usr/bin/env bash
# Evaluate a trained checkpoint and run modality ablations (ECG-only, EHR-only).
#
# NOTE: --is_training 0 loads <checkpoints>/<setting>/checkpoint.pth and evaluates
# it. If that file does not exist the run now fails loudly instead of silently
# scoring a randomly initialised model.
#
# NOTE: CLI flags override the YAML (run.py::resolve_args). That is what makes the
# ablations below real ablations -- and what keeps each --setting writing to its
# own checkpoint/results directory instead of all of them clobbering the setting
# name declared inside ckd_prognosis.yaml.
set -euo pipefail
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

# evaluate the trained multimodal model (no training)
python run.py --is_training 0 --config configs/ckd_prognosis.yaml \
  --setting ckd_2y_survival_multimodal

# ablation: ECG only
python run.py --is_training 1 --config configs/ckd_prognosis.yaml \
  --setting ckd_2y_ecg_only --use_ecg 1 --use_ehr 0

# ablation: EHR only
python run.py --is_training 1 --config configs/ckd_prognosis.yaml \
  --setting ckd_2y_ehr_only --use_ecg 0 --use_ehr 1

# ablation: fixed-horizon binary head instead of survival
python run.py --is_training 1 --config configs/ckd_prognosis.yaml \
  --setting ckd_2y_binary --head binary

# ablation: cross-attention fusion over the full patch sequence
python run.py --is_training 1 --config configs/ckd_prognosis.yaml \
  --setting ckd_2y_cross_patches --fusion cross --cross_attn_tokens patches

# ablation: cross-attention over pooled granularity tokens only (legacy behaviour)
python run.py --is_training 1 --config configs/ckd_prognosis.yaml \
  --setting ckd_2y_cross_granularity --fusion cross --cross_attn_tokens granularity
