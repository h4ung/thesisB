"""Entry point for CKD prognosis experiments.

Example
-------
python run.py \
    --task_name ckd_prognosis --is_training 1 \
    --root_path dataset/ckd --data MIMICIV_CKD \
    --model CardioformerCKD \
    --enc_in 12 --seq_len 250 \
    --d_model 128 --n_heads 8 --e_layers 6 --d_ff 256 \
    --patch_len_list 2,4,8,8,16,16,16,16,32 \
    --head survival --n_intervals 8 --fusion gated \
    --use_ecg 1 --use_ehr 1 \
    --batch_size 16 --learning_rate 1e-4 --train_epochs 20 --patience 5

Each parameter is documented inline below. A YAML config can also be supplied
via --config; CLI flags override YAML values.
"""

import argparse
import os

import yaml

from exp.exp_ckd_prognosis import Exp_CKD_Prognosis


def str2bool(v):
    return str(v).lower() in ("1", "true", "yes", "y")


def build_parser():
    p = argparse.ArgumentParser(description="CardioformerCKD — prognostic CKD prediction")

    # basic
    p.add_argument("--task_name", default="ckd_prognosis")
    p.add_argument("--is_training", type=int, default=1)
    p.add_argument("--model", default="CardioformerCKD")
    p.add_argument("--setting", default="ckd_run")
    p.add_argument("--config", default=None, help="optional YAML config path")

    # data
    p.add_argument("--root_path", default="dataset/ckd")
    p.add_argument("--data", default="MIMICIV_CKD")
    p.add_argument("--seq_len", type=int, default=250, help="ECG window length (timestamps)")
    p.add_argument("--enc_in", type=int, default=12, help="number of ECG leads")
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--augment", type=str2bool, default=True)

    # Cardioformer backbone
    p.add_argument("--d_model", type=int, default=128)
    p.add_argument("--n_heads", type=int, default=8)
    p.add_argument("--e_layers", type=int, default=6)
    p.add_argument("--d_ff", type=int, default=256)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--patch_len_list", default="2,4,8,8,16,16,16,16,32")
    p.add_argument("--resnet_hidden", type=int, default=64)
    p.add_argument("--resnet_blocks", type=int, default=2)
    p.add_argument("--cross_channel", type=str2bool, default=True)

    # multimodal / prognostic
    p.add_argument("--use_ecg", type=str2bool, default=True)
    p.add_argument("--use_ehr", type=str2bool, default=True)
    p.add_argument("--ehr_mode", choices=["mlp", "seq"], default="mlp")
    p.add_argument("--ehr_dropout", type=float, default=0.2)
    p.add_argument("--fusion", choices=["concat", "gated", "cross"], default="gated")
    p.add_argument("--d_fuse", type=int, default=256)
    p.add_argument("--head", choices=["binary", "survival"], default="survival")
    p.add_argument("--n_intervals", type=int, default=8, help="survival time bins over horizon")

    # optimisation
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--learning_rate", type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--train_epochs", type=int, default=20)
    p.add_argument("--patience", type=int, default=5)
    p.add_argument("--lradj", default="type1")
    p.add_argument("--focal_alpha", type=float, default=0.25)
    p.add_argument("--focal_gamma", type=float, default=2.0)
    p.add_argument("--seed", type=int, default=41)

    # system
    p.add_argument("--use_gpu", type=str2bool, default=True)
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--devices", default="0")
    p.add_argument("--checkpoints", default="checkpoints/ckd_prognosis")
    p.add_argument("--results", default="results/ckd_prognosis")
    return p


def apply_yaml(args):
    if args.config and os.path.exists(args.config):
        cfg = yaml.safe_load(open(args.config))
        defaults = {k: v for k, v in cfg.items() if hasattr(args, k)}
        # only set values the user did not explicitly pass on the CLI
        for k, v in defaults.items():
            setattr(args, k, v)
    return args


def main():
    args = build_parser().parse_args()
    args = apply_yaml(args)

    os.environ.setdefault("CUDA_VISIBLE_DEVICES", args.devices)
    print("Args:", vars(args))

    exp = Exp_CKD_Prognosis(args)
    if args.is_training:
        exp.train()
    exp.test()


if __name__ == "__main__":
    main()
