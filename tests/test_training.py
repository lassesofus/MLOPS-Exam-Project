import os
from datetime import datetime

import numpy as np
import pytest
import torch
import random
from torch.utils.data import random_split
from hydra import compose, initialize
from torch.nn import BCEWithLogitsLoss
from torch.optim import Adam
from torch.utils.data import DataLoader

from src.data.data_utils import load_dataset
from src.models.model import BERT
from src.models.train_model import eval, train, train_epoch


@pytest.mark.skipif(not os.path.exists("./data"),
                    reason="Data files not found")
def test_train_epoch() -> None:
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
    device = "cpu"
    model = BERT(drop_p=0.5).to(device)
    optimizer = Adam(params=model.parameters(), lr=1e-05)
    criterion = BCEWithLogitsLoss()
    result = train_epoch(model, criterion, optimizer,
                         train_loader, epochs, device)

    assert (
        np.size(result) == 1
    ), "The dimension of the returned object of 'train_epoch()'\
        is not as expected!"


@pytest.mark.skipif(not os.path.exists("./data"),
                    reason="Data files not found")
def test_train() -> None:
    with initialize(version_base=None, config_path="conf_test"):
        cfg = compose(config_name="config.yaml")
    data_part = load_dataset(cfg, cfg.train.path_train_set)
    train_set, val_set = random_split(dataset=data_part, lengths=[0.9, 0.1])
    subset = list(range(0, 8))
    trainset_subset = torch.utils.data.Subset(train_set, subset)
    valset_subset = torch.utils.data.Subset(val_set, subset)

    train_loader = DataLoader(trainset_subset, batch_size=16, shuffle=True)
    val_loader = DataLoader(valset_subset, batch_size=cfg.train.batch_size,
                            shuffle=False)

    debug_mode = True
    device = "cpu"
    model = BERT(drop_p=0.5).to(device)
    optimizer = Adam(params=model.parameters(), lr=1e-05)
    criterion = BCEWithLogitsLoss()

    time = datetime.now().strftime("%H-%M-%S")
    result = train(cfg, model, criterion, optimizer,
                   train_loader, val_loader, device, debug_mode)

    assert (
        result == f"./models/T{time}.pt"
    ), "The returned path of 'train()' is not as expected!"


@pytest.mark.skipif(not os.path.exists("./data"),
                    reason="Data files not found")
def test_eval() -> None:
    with initialize(version_base=None, config_path="conf_test"):
        cfg = compose(config_name="config.yaml")
    path_test = "data/processed/test.csv"
    data_part = load_dataset(cfg, cfg.train.path_train_set)
    train_set, val_set = random_split(dataset=data_part, lengths=[0.9, 0.1])
    
    test_set = load_dataset(cfg, path_test)
    subset = list(range(0, 8))
    trainset_subset = torch.utils.data.Subset(train_set, subset)
    testset_subset = torch.utils.data.Subset(test_set, subset)
    valset_subset = torch.utils.data.Subset(val_set, subset)

    train_loader = DataLoader(trainset_subset, batch_size=16, shuffle=True)
    test_loader = DataLoader(testset_subset, batch_size=16, shuffle=True)
    val_loader = DataLoader(valset_subset, batch_size=cfg.train.batch_size,
                            shuffle=False)
    debug_mode = True
    device = "cpu"
    model = BERT(drop_p=0.5).to(device)
    optimizer = Adam(params=model.parameters(), lr=1e-05)
    criterion = BCEWithLogitsLoss()

    weights = train(cfg, model, criterion, optimizer,
                    train_loader, val_loader, device, debug_mode)
    result = eval(model, weights, criterion, test_loader, device, debug_mode)
    assert (
        np.size(result) == 1
    ), "The dimension of the returned object of 'test()' \
        is not as expected!"
