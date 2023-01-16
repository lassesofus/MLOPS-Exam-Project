from datetime import datetime

import hydra
import matplotlib.pyplot as plt
import numpy as np
import torch
from dotenv import find_dotenv, load_dotenv
from omegaconf import DictConfig
from sklearn import metrics
from torch import nn
from torch.nn import BCEWithLogitsLoss
from torch.optim import Adam
from torch.utils.data import DataLoader
from tqdm import tqdm
import wandb 

import wandb
from src.data.data_utils import load_dataset
from src.models.model import BERT


def train_epoch(
    model: nn.Module,
    criterion: BCEWithLogitsLoss,
    optimizer: Adam,
    train_loader: DataLoader,
    epoch: int,
    device: torch.cuda.device,
) -> float: 
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
            temp = data["token_type_ids"]
            token_type_ids = temp.to(device, dtype=torch.long)
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
    cfg: DictConfig,
    model: nn.Module,
    criterion: BCEWithLogitsLoss,
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
    time = datetime.now().strftime("%H-%M-%S")

    for epoch in range(cfg.train.hps.epochs):
        # Train 1 epoch
        epoch_loss = train_epoch(
            model, criterion, optimizer, train_loader, epoch, device
        )
        epoch_losses.append(epoch_loss)

        # Save wandb log
        wandb.log({
            "training_loss": epoch_loss,
            "epoch": epoch
        })

        # Save if model is better
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            save_path = f"./models/T{time}_E{epoch+1}.pt"
            torch.save(model.state_dict(), save_path)

    # Plot graph of training  loss
    plt.figure()
    plt.plot(epoch_losses, label="Training loss")
    plt.legend()
    plt.savefig("./reports/figures/loss.png")

    # Print best best loss
    print(f"Best loss: {best_loss}")

    return save_path


def test(
    model: nn.Module,
    weights: str,
    criterion: BCEWithLogitsLoss,
    test_loader: DataLoader,
    device: torch.cuda.device
) -> None: 
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
    batch_losses = []

    with torch.no_grad():
        with tqdm(test_loader, desc="Test epoch") as tepoch:
            for _, data in enumerate(tepoch):
                # Extracting data from the data batch
                ids = data["ids"].to(device, dtype=torch.long)
                mask = data["mask"].to(device, dtype=torch.long)
                temp = data["token_type_ids"]
                token_type_ids = temp.to(device, dtype=torch.long)
                targets = data["targets"].to(device, dtype=torch.float)

                # Running the model on the data to get the outputs
                outputs = model(ids, mask, token_type_ids)

                # Appending the targets and outputs to lists
                # (apply sigmoid to logits)
                fin_targets.extend(
                    targets.cpu().detach().numpy().tolist()
                )
                fin_outputs.extend(
                    torch.sigmoid(outputs).cpu().detach().numpy().tolist()
                )

    # Map output probs to labels (get predictions)
    fin_outputs = np.array(fin_outputs) >= 0.5


    # Calculate accuracy and f1 score
    # TODO: Add confusion matrix visualization here or
    # in the cookie-cutter directory
    accuracy = metrics.accuracy_score(fin_targets, fin_outputs)
    f1_score_micro = metrics.f1_score(fin_targets, fin_outputs,
                                      average="micro")
    f1_score_macro = metrics.f1_score(fin_targets, fin_outputs,

                                      average="macro")
    print(f"Accuracy Score = {accuracy}")
    print(f"F1 Score (Micro) = {f1_score_micro}")
    print(f"F1 Score (Macro) = {f1_score_macro}")

    return accuracy


@hydra.main(version_base=None, config_name="config.yaml", config_path="conf")
def main(cfg: DictConfig) -> None:
    # Fetch secret environment variables 
    dotenv_path = find_dotenv()
    load_dotenv(dotenv_path)

    # Initialize wandb
    wandb.init(config=cfg, project="test-project", entity="louisdt")

    # Set up hyper-parameters # TODO: What to do with these?
    #device = "cuda" if torch.cuda.is_available() else "cpu"
    device = cfg.training.hyperparameters.device
    path_train = cfg.training.hyperparameters.path_train
    path_test = cfg.training.hyperparameters.path_test

    # Load training data
    train_set = load_dataset(path_train)
    test_set = load_dataset(path_test)

    train_loader = DataLoader(
        train_set, batch_size=cfg.training.hyperparameters.train_batch_size,
        shuffle=True
    )
    test_loader = DataLoader(
        test_set, batch_size=cfg.training.hyperparameters.valid_batch_size,
        shuffle=False
    )

    # Initialize model, objective and optimizer
    model = BERT(drop_p=cfg.model.hyperparameters.drop_p,
                 embed_dim=cfg.model.hyperparameters.embed_dim,
                 out_dim=cfg.model.hyperparameters.out_dim).to(device)

    criterion = BCEWithLogitsLoss()
    optimizer = Adam(params=model.parameters(),
                     lr=cfg.training.hyperparameters.learning_rate)

    # Train model
    weights = train(cfg, model, criterion, optimizer, train_loader, device)

    # Test model
    test(model, weights, test_loader, device)


if __name__ == "__main__":
    main()
