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
import random
import os
from torch.utils.data import random_split

import wandb
from src.data.data_utils import load_dataset
from src.models.model import BERT


def train_epoch(
    model: nn.Module,
    criterion: BCEWithLogitsLoss,
    optimizer: Adam,
    train_loader: DataLoader,
    epoch: int,
    device: torch.device,
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
            # Clean
            optimizer.zero_grad()
            
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
            loss.backward()
            optimizer.step()

            # Save loss
            batch_losses.append(loss.item())

    print("Training_Loss: ", np.mean(batch_losses))
    return np.mean(batch_losses)


def val_epoch(
    model: nn.Module,
    criterion: BCEWithLogitsLoss,
    val_loader: DataLoader,
    epoch: int,
    device: torch.device,
) -> float:

    """
    Validate model for a single epoch

    :param model: Model to train
    :param criterion: Loss function
    :param optimizer: Optimizer
    :param train_loader: Training data loader
    :param device: Device to train on
    :param epoch: Current epoch
    :return: Mean loss for epoch
    """

    model.eval()
    with torch.no_grad():
        batch_losses = []

        # Iterate over training data
        with tqdm(val_loader, desc=f"Validation epoch {epoch}") as tepoch:
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

                # Save loss
                batch_losses.append(loss.item())

        # Epoch loss
        epoch_loss = np.mean(batch_losses)

        print("Validation_Loss: ", epoch_loss)

    return epoch_loss


def train(
    cfg: DictConfig,
    model: nn.Module,
    criterion: BCEWithLogitsLoss,
    optimizer: Adam,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    debug_mode: bool = False,
) -> str:
    """
    Trains the model for a given amount of epochs

    :param cfg: Hydra config
    :param model: Model to train
    :param criterion: Loss function
    :param optimizer: Optimizer
    :param train_loader: Training data loader
    :param val_loader: Validation data loader
    :param device: Device to train on
    :param debug_mode: Specify whether it should be logged or not
    :returns: File path to the saved model weights
    """

    train_losses = []
    val_losses = []
    best_loss = float("inf")
    time = datetime.now().strftime("%H-%M-%S")
    best_epoch = -1

    for epoch in range(cfg.train.epochs):
        # Train and validate 1 epoch
        train_loss = train_epoch(
            model, criterion, optimizer, train_loader, epoch, device
        )
        val_loss = val_epoch(
            model, criterion, val_loader, epoch, device
        )

        # Save epoch loss and wandb log
        train_losses.append(train_loss)
        val_losses.append(val_loss)

        if debug_mode is False:
            wandb.log({
                "train_loss": train_loss,
                "val_loss": val_loss,
                "epoch": epoch
            })

        # Save if model is better
        if val_loss < best_loss:
            best_loss = val_loss
            save_path = f"./models/T{time}.pt"
            torch.save(model.state_dict(), save_path)
            best_epoch = epoch + 1

    print('Best model trained for '+str(best_epoch)+' epochs')

    # Plot graph of training  loss
    plt.figure()
    plt.plot(train_losses, label="Training loss")
    plt.plot(val_losses, label="Validation loss")
    plt.legend()
    plt.savefig("./reports/figures/loss.png")

    # Print best best loss
    print(f"Best validation loss: {best_loss}")

    if debug_mode is False:
        wandb.log({
            "Model_Name": save_path,
            "best_epoch": best_epoch
        })

    return save_path


def eval(
    model: nn.Module,
    weights: str,
    criterion: BCEWithLogitsLoss,
    test_loader: DataLoader,
    device: torch.device,
    debug_mode: bool = False,
) -> None:
    """
    Run model on the test set
    :param model: Initialized model
    :param weights: File path to the saved model weights
    :param test_loader: Test data loader
    :param device: Device to train on
    :param debug_mode: Specify whether it should be logged or not
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
                loss = criterion(outputs, targets)

                # Appending the targets and outputs to lists
                # (apply sigmoid to logits)
                fin_targets.extend(
                    targets.cpu().detach().numpy().tolist()
                )
                fin_outputs.extend(
                    torch.sigmoid(outputs).cpu().detach().numpy().tolist()
                )

                # Save loss
                batch_losses.append(loss.item())

    # Map output probs to labels (get predictions)
    fin_outputs = np.array(fin_outputs) >= 0.5

    # Calculate mean loss and metrics
    epoch_loss = np.mean(batch_losses)
    accuracy = metrics.accuracy_score(fin_targets, fin_outputs)
    f1_score_micro = metrics.f1_score(fin_targets, fin_outputs,
                                      average="micro")
    f1_score_macro = metrics.f1_score(fin_targets, fin_outputs,
                                      average="macro")

    # Print and wandb log metrics
    if debug_mode is False:
        wandb.log({
            "test_loss": epoch_loss,
            "accuracy": accuracy,
            "f1_score_micro": f1_score_micro,
            "f1_score_macro": f1_score_macro
        })

    print(f"Loss = {epoch_loss}")
    print(f"Accuracy Score = {accuracy}")
    print(f"F1 Score (Micro) = {f1_score_micro}")
    print(f"F1 Score (Macro) = {f1_score_macro}")

    return accuracy


@hydra.main(version_base=None, 
            config_name="config.yaml", 
            config_path="../../hydra_config")
def main(cfg: DictConfig) -> None:
    """ Run training and test, save best model and log metrics

    :param cfg: Hydra config
    """

    # Set random seed
    seed = cfg.train.seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # Set device
    if cfg.train.gpu_override == 1:
        device = torch.device("cpu")
        print('Using CPU override!')
    else:
        if torch.cuda.is_available():
            device = torch.device("cuda")
            print('Using available GPU!')
        else:
            device = torch.device("cpu")
            print('Using available CPU!')

    # Fetch wand authorization (working with cloned/forked repo
    # requires wandb key to be defined in .env file and running
    # training docker image requires wandb key to be definned as
    # environment variable with flag -e WANDB_API_KEY=... when
    # calling docker run)
    check_env1 = ("WANDB_API_KEY" not in os.environ)
    check_env2 = ("WANDB_ENTITY" not in os.environ)
    check_env3 = ("WANDB_PROJECT" not in os.environ)

    if check_env1 or check_env2 or check_env3:
        dotenv_path = find_dotenv()
        load_dotenv(dotenv_path)

    # Initialize wandb
    wandb.init(config=cfg)

    # Load training data
    data_part = load_dataset(cfg, cfg.train.path_train_set)
    train_set, val_set = random_split(dataset=data_part, lengths=[0.9, 0.1])
    test_set = load_dataset(cfg, cfg.train.path_test_set)

    train_loader = DataLoader(
        train_set, batch_size=cfg.train.batch_size,
        shuffle=True
    )
    val_loader = DataLoader(
        val_set, batch_size=cfg.train.batch_size,
        shuffle=False
    )
    test_loader = DataLoader(
        test_set, batch_size=cfg.train.batch_size,
        shuffle=False
    )

    # Initialize model, objective and optimizer
    model = BERT(
        drop_p=cfg.model.drop_p,
    ).to(device)

    criterion = BCEWithLogitsLoss()
    optimizer = Adam(params=model.parameters(),
                     lr=cfg.train.learning_rate)

    # Train model
    weights = train(cfg, model, criterion, optimizer,
                    train_loader, val_loader, device)

    # Test model
    _ = eval(model, weights, criterion, test_loader, device)


if __name__ == "__main__":
    main()
