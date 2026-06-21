"""Base experiment class (Time-Series-Library style)."""

import torch


class Exp_Basic:
    def __init__(self, args):
        self.args = args
        self.device = self._acquire_device()
        self.model = self._build_model().to(self.device)

    def _build_model(self):
        raise NotImplementedError

    def _acquire_device(self):
        if self.args.use_gpu and torch.cuda.is_available():
            dev = torch.device(f"cuda:{self.args.gpu}")
            print(f"Use GPU: cuda:{self.args.gpu}")
        else:
            dev = torch.device("cpu")
            print("Use CPU")
        return dev

    def train(self):
        raise NotImplementedError

    def test(self):
        raise NotImplementedError
