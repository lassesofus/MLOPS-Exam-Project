# -*- coding: utf-8 -*-
#!/usr/bin/python
import hydra
import torch
from tqdm import tqdm
import numpy as np
from torch.optim import Adam
from torch.nn import BCEWithLogitsLoss
from datetime import datetime
import matplotlib.pyplot as plt
from torch import nn
from sklearn import metrics

from torch.utils.data import DataLoader
from src.models.model import BERT
from src.data.data_utils import load_dataset


def train_epoch(
    model: nn.Module,
    criterion: BCEWithLogitsLoss,
    optimizer: Adam,
    train_loader: DataLoader,
    epoch: int,
    device: torch.cuda.device,
) -> float:  # TODO: float or some numpy object?
    """
    Train model for a single epoch

    :param model: Model to train
    :param criterion: Loss function
    :param optimizer: Optimizer
    :param train_loader: Training data loader
    :param epoch: Current epoch
    :param device: Device to train on
    :return: Mean loss for epoch
    """

    model.train()
    batch_losses = []

    # Iterate over training data
    with tqdm(train_loader, desc=f"Epoch {epoch}") as tepoch:
        for _, data in enumerate(tepoch):
            # Move data to device
            ids = data["ids"].to(device, dtype=torch.long)
            mask = data["mask"].to(device, dtype=torch.long)
            token_type_ids = data["token_type_ids"].to(device, dtype=torch.long)
            targets = data["targets"].to(device, dtype=torch.float)

            # Forward pass and loss calculation
            outputs = model(ids, mask, token_type_ids)
            loss = criterion(outputs, targets)

            # Backpropagate and update weights
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Save loss
            batch_losses.append(loss.item())

    return np.mean(batch_losses)


def train(
    cfg,
    model: nn.Module,
    criterion: BCEWithLogitsLoss,  # TODO: Add typing for hydra cfg
    optimizer: Adam,
    train_loader: DataLoader,
    device: torch.cuda.device,
) -> str:
    """
    Trains the model for a given amount of epochs

    :param cfg: Hydra config
    :param model: Model to train
    :param criterion: Loss function
    :param optimizer: Optimizer
    :param train_loader: Training data loader
    :param device: Device to train on
    :returns: File path to the saved model weights
    """

    epoch_losses = []
    best_loss = float("inf")
    start_time = datetime.now().strftime("%H-%M-%S")

    for epoch in range(cfg.hps.epochs):
        # Train 1 epoch
        epoch_loss = train_epoch(
            model, criterion, optimizer, train_loader, epoch, device
        )
        epoch_losses.append(epoch_loss)

        # Save if model is better
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            save_path = f"./models/{start_time}.pt"
            torch.save(model.state_dict(), save_path)

    # Plot grap of training  loss
    plt.figure()
    plt.plot(epoch_losses, label="Training loss")
    plt.legend()
    plt.savefig("./reports/figures/loss.png")

    return save_path


def test(
    model: nn.Module, weights: str, test_loader: DataLoader, device: torch.cuda.device
) -> None:  # TODO: Add typing for hydra cfg
    """
    Run model on the test set

    :param model: Initialized model
    :param weights: File path to the saved model weights
    :param test_loader: Test data loader
    :param device: Device to train on
    """
    model.eval()

    # Load best model
    model.load_state_dict(torch.load(weights))

    # Test model
    fin_targets = []
    fin_outputs = []

    with torch.no_grad():
        with tqdm(test_loader, desc=f"Test epoch") as tepoch:
            for _, data in enumerate(tepoch):
                # Extracting data from the data batch
                ids = data["ids"].to(device, dtype=torch.long)
                mask = data["mask"].to(device, dtype=torch.long)
                token_type_ids = data["token_type_ids"].to(device, dtype=torch.long)
                targets = data["targets"].to(device, dtype=torch.float)

                # Running the model on the data to get the outputs
                outputs = model(ids, mask, token_type_ids)

                # Appending the targets and outputs to lists (apply sigmoid to logits)
                fin_targets.extend(targets.cpu().detach().numpy().tolist())
                fin_outputs.extend(
                    torch.sigmoid(outputs).cpu().detach().numpy().tolist()
                )

    # Map output probs to labels (get predictions)
    fin_outputs = np.array(fin_outputs) >= 0.5

    # Calculate accuracy and f1 score # TODO: Add confusion matrix visualization here or in the cookie-cutter directory
    accuracy = metrics.balanced_accuracy_score(fin_targets, fin_outputs)
    f1_score_micro = metrics.f1_score(fin_targets, fin_outputs, average="micro")
    f1_score_macro = metrics.f1_score(fin_targets, fin_outputs, average="macro")
    print(f"Accuracy Score = {accuracy}")
    print(f"F1 Score (Micro) = {f1_score_micro}")
    print(f"F1 Score (Macro) = {f1_score_macro}")


@hydra.main(version_base=None, config_name="config.yaml", config_path=".")
def main(cfg) -> None:  # TODO: Add typing for hydra cfg
    # Set up hyper-parameters # TODO: What to do with these?
    device = "cuda" if torch.cuda.is_available() else "cpu"
    path_train = "data/processed/train.csv"
    path_test = "data/processed/test.csv"  # TODO: Add to config

    # Load training data
    train_set = load_dataset(path_train)
    test_set = load_dataset(path_test)

    train_loader = DataLoader(
        train_set, batch_size=cfg.hps.train_batch_size, shuffle=True
    )
    test_loader = DataLoader(
        test_set, batch_size=cfg.hps.valid_batch_size, shuffle=False
    )

    # Initialize model, objective and optimizer
    model = BERT(drop_p=cfg.hps.drop_p).to(device)
    criterion = BCEWithLogitsLoss()
    optimizer = Adam(params=model.parameters(), lr=cfg.hps.learning_rate)

    # Train model
    weights = train(cfg, model, criterion, optimizer, train_loader, device)

    # Test model
    test(model, weights, test_loader, device)


if __name__ == "__main__":
    main()
