import os
from datetime import datetime

import numpy as np
import pytest
import torch
from torch.nn import BCEWithLogitsLoss
from torch.optim import Adam
from torch.utils.data import DataLoader

from src.data.data_utils import load_dataset
from src.models.model import BERT
from src.models.train_model import train, train_epoch, ttest


@pytest.mark.skipif(not os.path.exists("./data"),
                    reason="Data files not found")
def test_train_epoch() -> None:
    path_train = "data/processed/train.csv"
    train_set = load_dataset(path_train)
    subset = list(range(0, 8))
    trainset_subset = torch.utils.data.Subset(train_set, subset)

    train_loader = DataLoader(trainset_subset, batch_size=16, shuffle=True)
    epochs = 1
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = BERT(drop_p=0.5, embed_dim=768, out_dim=2).to(device)
    optimizer = Adam(params=model.parameters(), lr=1e-05)
    criterion = BCEWithLogitsLoss()
    result = train_epoch(
        model, criterion, optimizer, train_loader, epochs, device, batch_size=8
    )
    assert (
        np.size(result) == 1
    ), "The dimension of the returned object of 'train_epoch()'\
        is not as expected!"


@pytest.mark.skipif(not os.path.exists("./data"), reason="Data files not found")
def test_train() -> None:
    # Paths for data
    path_train = "data/processed/train.csv"
    # Load train datasets
    train_set = load_dataset(path_train)

    subset = list(range(0, 8))
    trainset_subset = torch.utils.data.Subset(train_set, subset)

    train_loader = DataLoader(trainset_subset, batch_size=16, shuffle=True)
    epochs = 1
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = BERT(drop_p=0.5, embed_dim=768, out_dim=2).to(device)
    optimizer = Adam(params=model.parameters(), lr=1e-05)
    criterion = BCEWithLogitsLoss()

    start_time = datetime.now().strftime("%H-%M-%S")
    print("works so far")
    result = train(
        epochs, model, criterion, optimizer, train_loader, device, batch_size=8
    )
    assert (
        result == f"./models/{start_time}.pt"
    ), "The returned path of 'train()' is not as expected!"


@pytest.mark.skipif(not os.path.exists("./data"), reason="Data files not found")
def test_test_func() -> None:
    path_train = "data/processed/train.csv"
    path_test = "data/processed/test.csv"
    train_set = load_dataset(path_train)
    test_set = load_dataset(path_test)
    subset = list(range(0, 8))
    trainset_subset = torch.utils.data.Subset(train_set, subset)
    testset_subset = torch.utils.data.Subset(test_set, subset)

    train_loader = DataLoader(trainset_subset, batch_size=16, shuffle=True)
    test_loader = DataLoader(testset_subset, batch_size=16, shuffle=True)
    epochs = 1
    batch_size = 8
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = BERT(drop_p=0.5, embed_dim=768, out_dim=2).to(device)
    optimizer = Adam(params=model.parameters(), lr=1e-05)
    criterion = BCEWithLogitsLoss()

    weights = train(
        epochs, model, criterion, optimizer, train_loader, device, batch_size
    )
    result = ttest(model, weights, test_loader, device, batch_size)

    assert (
        np.size(result) == 1
    ), "The dimension of the returned object of 'test()' \
        is not as expected!"
