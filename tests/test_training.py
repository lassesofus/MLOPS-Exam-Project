import os
from datetime import datetime

import numpy as np
import pytest
import torch
from torch.nn import BCEWithLogitsLoss
from torch.optim import Adam
from torch.utils.data import DataLoader
from hydra import compose, initialize

from src.data.data_utils import load_dataset
from src.models.model import BERT
from src.models.train_model import train, train_epoch, eval


@pytest.mark.skipif(not os.path.exists("./data"),
                    reason="Data files not found")
def test_train_epoch() -> None:
    with initialize(version_base=None, config_path="conf_test"):
        cfg = compose(config_name="config.yaml")
    path_train = "data/processed/train.csv"
    train_set = load_dataset(cfg, path_train)
    subset = list(range(0, 8))
    trainset_subset = torch.utils.data.Subset(train_set, subset)

    train_loader = DataLoader(trainset_subset, batch_size=16, shuffle=True)
    epochs = 1
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = BERT(drop_p=0.5, embed_dim=768, out_dim=2).to(device)
    optimizer = Adam(params=model.parameters(), lr=1e-05)
    criterion = BCEWithLogitsLoss()
    result = train_epoch(
        cfg, model, criterion, optimizer, train_loader, epochs
    )
    assert (
        np.size(result) == 1
    ), "The dimension of the returned object of 'train_epoch()'\
        is not as expected!"


@pytest.mark.skipif(not os.path.exists("./data"), reason="Data files not found")
def test_train() -> None:
    with initialize(version_base=None, config_path="conf_test"):
        cfg = compose(config_name="config.yaml")
    # Paths for data
    path_train = "data/processed/train.csv"
    # Load train datasets
    train_set = load_dataset(cfg, path_train)

    subset = list(range(0, 8))
    trainset_subset = torch.utils.data.Subset(train_set, subset)

    train_loader = DataLoader(trainset_subset, batch_size=16, shuffle=True)
    epochs = 1
    debug_mode = True
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = BERT(drop_p=0.5, embed_dim=768, out_dim=2).to(device)
    optimizer = Adam(params=model.parameters(), lr=1e-05)
    criterion = BCEWithLogitsLoss()

    time = datetime.now().strftime("%H-%M-%S")
    result = train(
        cfg, model, criterion, optimizer, train_loader, debug_mode
    )

    assert (
        result == f"./models/T{time}_E{1}.pt"
    ), "The returned path of 'train()' is not as expected!"


@pytest.mark.skipif(not os.path.exists("./data"), reason="Data files not found")
def test_eval() -> None:
    with initialize(version_base=None, config_path="conf_test"):
        cfg = compose(config_name="config.yaml")
    path_train = "data/processed/train.csv"
    path_test = "data/processed/test.csv"
    train_set = load_dataset(cfg, path_train)
    test_set = load_dataset(cfg, path_test)
    subset = list(range(0, 8))
    trainset_subset = torch.utils.data.Subset(train_set, subset)
    testset_subset = torch.utils.data.Subset(test_set, subset)

    train_loader = DataLoader(trainset_subset, batch_size=16, shuffle=True)
    test_loader = DataLoader(testset_subset, batch_size=16, shuffle=True)
    epochs = 1
    batch_size = 8
    debug_mode = True
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = BERT(drop_p=0.5, embed_dim=768, out_dim=2).to(device)
    optimizer = Adam(params=model.parameters(), lr=1e-05)
    criterion = BCEWithLogitsLoss()

    weights = train(
        cfg, model, criterion, optimizer, train_loader, debug_mode
    )
    result = eval(cfg, model, weights, criterion, test_loader, debug_mode)
    assert (
        np.size(result) == 1
    ), "The dimension of the returned object of 'test()' \
        is not as expected!"

