import math
import torch
from torch.utils.data import DataLoader, Dataset
from torch.nn import BCEWithLogitsLoss
from torch import nn
from torch.optim import Adam

from src.models.train_model import train_epoch, val_epoch


class DummyDataset(Dataset):
    def __init__(self):
        self.samples = [
            {
                "ids": torch.tensor([1, 2]),
                "mask": torch.tensor([1, 1]),
                "token_type_ids": torch.tensor([0, 0]),
                "targets": torch.tensor([1.0, 0.0]),
            },
            {
                "ids": torch.tensor([3, 4]),
                "mask": torch.tensor([1, 1]),
                "token_type_ids": torch.tensor([0, 0]),
                "targets": torch.tensor([0.0, 1.0]),
            },
        ]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


class ConstantModel(nn.Module):
    def __init__(self):
        super().__init__()
        # single parameter so optimizer does not raise an error
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, ids, mask, token_type_ids):
        batch_size = ids.size(0)
        return torch.zeros(batch_size, 2) + self.bias


def _setup():
    dataset = DummyDataset()
    loader = DataLoader(dataset, batch_size=2)
    model = ConstantModel()
    criterion = BCEWithLogitsLoss()
    optimizer = Adam(model.parameters(), lr=0.0)
    return model, criterion, optimizer, loader


def test_train_epoch_constant_loss():
    model, criterion, optimizer, loader = _setup()
    loss = train_epoch(model, criterion, optimizer, loader, epoch=0, device="cpu")
    assert math.isclose(loss, math.log(2.0), rel_tol=1e-6)


def test_val_epoch_constant_loss():
    model, criterion, optimizer, loader = _setup()
    loss = val_epoch(model, criterion, loader, epoch=0, device="cpu")
    assert math.isclose(loss, math.log(2.0), rel_tol=1e-6)
