import torch

from models.CardioformerCKD import CardioformerCKD
from models.heads import DiscreteTimeSurvivalHead


class Cfg:
    # backbone
    task_name = "representation"
    enc_in = 12
    seq_len = 250
    d_model = 32
    n_heads = 4
    e_layers = 2
    d_ff = 64
    dropout = 0.1
    patch_len_list = "8,16,32"
    resnet_hidden = 16
    resnet_blocks = 1
    cross_channel = True
    # multimodal
    use_ecg = True
    use_ehr = True
    ehr_in_dim = 20
    ehr_mode = "mlp"
    ehr_dropout = 0.2
    fusion = "gated"
    d_fuse = 64
    head = "survival"
    n_intervals = 8


def _batch(B=4, cfg=Cfg):
    x_ecg = torch.randn(B, cfg.seq_len, cfg.enc_in)
    ehr = torch.randn(B, cfg.ehr_in_dim)
    return x_ecg, ehr


def test_survival_forward_shape():
    cfg = Cfg()
    model = CardioformerCKD(cfg)
    x_ecg, ehr = _batch()
    out = model(x_ecg=x_ecg, ehr=ehr)
    assert out.shape == (4, cfg.n_intervals)
    surv, cif = DiscreteTimeSurvivalHead.survival_from_hazards(out)
    assert surv.shape == cif.shape == (4, cfg.n_intervals)
    # survival is monotonically non-increasing
    assert torch.all(surv[:, 1:] <= surv[:, :-1] + 1e-5)


def test_binary_head_and_fusions():
    for fusion in ["concat", "gated", "cross"]:
        cfg = Cfg(); cfg.head = "binary"; cfg.fusion = fusion
        model = CardioformerCKD(cfg)
        x_ecg, ehr = _batch()
        out = model(x_ecg=x_ecg, ehr=ehr)
        assert out.shape == (4,)


def test_unimodal_paths():
    # ECG only
    cfg = Cfg(); cfg.use_ehr = False; cfg.head = "binary"
    m = CardioformerCKD(cfg)
    x_ecg, _ = _batch()
    assert m(x_ecg=x_ecg).shape == (4,)
    # EHR only
    cfg2 = Cfg(); cfg2.use_ecg = False; cfg2.head = "binary"
    m2 = CardioformerCKD(cfg2)
    _, ehr = _batch()
    assert m2(ehr=ehr).shape == (4,)


def test_backward_pass():
    cfg = Cfg()
    model = CardioformerCKD(cfg)
    x_ecg, ehr = _batch()
    out = model(x_ecg=x_ecg, ehr=ehr)
    loss = out.pow(2).mean()
    loss.backward()
    grads = [p.grad is not None for p in model.parameters() if p.requires_grad]
    assert any(grads)
