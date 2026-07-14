# Cardioformer-CKD: Prognostic Chronic Kidney Disease Prediction

Multimodal **prognostic** deep-learning model that predicts whether a patient
will develop **incident chronic kidney disease (CKD)** within a future horizon,
by combining the **MIMIC-IV** electronic health record (EHR) with the
**MIMIC-IV-ECG** 12-lead waveform dataset.

## Provenance of the ECG backbone (important for the thesis)

`models/Cardioformer.py`, `layers/Embed.py` and `layers/Cardioformer_EncDec.py`
are an **independent clean-room re-implementation** of the Cardioformer encoder,
written from the description in the paper (Mobin et al.,
*Cardioformer: Advancing AI in ECG Analysis with Multi-Granularity Patching and
ResNet*, [arXiv:2505.05538](https://arxiv.org/abs/2505.05538), 2025). **No code
from the authors' repository was used, and this implementation has not been
validated against their released weights or results.**

Therefore:

* published Cardioformer numbers are **not** directly comparable to numbers
  produced here — the ECG-only baseline should be reported as *"our
  re-implementation of Cardioformer"*, never as *"Cardioformer"*;
* details the paper leaves unspecified (patch strides, router-token design,
  normalisation placement, initialisation) are our own choices, documented in the
  module docstrings;
* a gap against the published results is expected and is a property of the
  re-implementation, not a failed reproduction.

## Configuration precedence

```
explicit CLI flag  >  --config YAML  >  argparse default
```

`run.py::resolve_args` only writes a YAML key into the namespace if that flag was
**not** typed on the command line. This is what makes

```bash
python run.py --config configs/ckd_prognosis.yaml --setting ckd_2y_ecg_only --use_ehr 0
```

a real ablation (and keeps it writing to its own checkpoint/results directory
rather than the `setting` declared inside the YAML).

## Evaluating a trained model

```bash
python run.py --is_training 0 --config configs/ckd_prognosis.yaml \
  --setting ckd_2y_survival_multimodal          # loads checkpoints/<setting>/checkpoint.pth
python run.py --is_training 0 --checkpoint_path path/to/checkpoint.pth ...   # or point at one
```

`test()` always loads the checkpoint from disk before scoring. A missing
checkpoint raises `FileNotFoundError` rather than silently evaluating randomly
initialised weights.

## Reported metrics

| metric | computed over |
|---|---|
| `c_index` | all patients (right-censoring handled by the comparable-pair definition) |
| `td_auroc` | **eligible patients only** — event by the horizon, or follow-up reaching it |
| `brier_at_horizon` | eligible patients only |
| `n_eligible` / `n_total` | how many patients the horizon metrics were computed on |

Patients censored *before* the horizon have an unknown horizon label; scoring
them as negatives inflates `td_auroc` (see `tests/test_metrics.py`).

See `PATCHES.md` for the full list of fixes and their effect on results.
