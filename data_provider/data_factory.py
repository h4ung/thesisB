"""Data factory (Time-Series-Library style).

Register new datasets here. ``data_dict`` maps the ``--data`` CLI name to a
Dataset class. The CKD prognosis task uses ``MIMICIV_CKD``.
"""

from torch.utils.data import DataLoader

from data_provider.data_loader import MIMICIV_CKD_Dataset

data_dict = {
    "MIMICIV_CKD": MIMICIV_CKD_Dataset,
}


def data_provider(args, flag, scaler=None, ehr_columns=None):
    Data = data_dict[args.data]
    shuffle = flag == "train"
    drop_last = flag == "train"

    dataset = Data(
        root_path=args.root_path,
        flag=flag,
        seq_len=args.seq_len,
        use_ecg=getattr(args, "use_ecg", True),
        use_ehr=getattr(args, "use_ehr", True),
        scaler=scaler,
        ehr_columns=ehr_columns,
        augment=getattr(args, "augment", True),
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=shuffle,
        num_workers=args.num_workers,
        drop_last=drop_last,
        pin_memory=True,
    )
    return dataset, loader
