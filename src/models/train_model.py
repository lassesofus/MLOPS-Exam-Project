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

from torch.utils.data import DataLoader
from src.models.model import BERT
from src.data.data_utils import load_dataset

def train_epoch(model:nn.Module, criterion:BCEWithLogitsLoss, 
                optimizer:Adam, train_loader:DataLoader, epoch:int, 
                device:torch.cuda.device) -> float:  # TODO: float or some numpy object?
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

#@hydra.main(version_base=None, config_name="config.yaml", config_path="conf")
def train(cfg, model:nn.Module, criterion:BCEWithLogitsLoss, # TODO: Add typing for hydra cfg
          optimizer:Adam, train_loader:DataLoader, 
          device:torch.cuda.device) -> None:
    """ 
    Trains the model for a given amount of epochs

    :param cfg: Hydra config
    :param model: Model to train
    :param criterion: Loss function
    :param optimizer: Optimizer
    :param train_loader: Training data loader
    :param device: Device to train on
    """

    epoch_losses = []
    best_loss = float("inf")
    start_time = datetime.now().strftime("%H-%M-%S")

    for epoch in range(cfg.training.hyperparameters.epochs):
        # Train 1 epoch 
        epoch_loss = train_epoch(model, criterion, optimizer, train_loader, epoch, device)
        epoch_losses.append(epoch_loss)

        # Save if model is better
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            torch.save(model.state_dict(), f"./models/{start_time}.pt")

    # Plot grap of training  loss
    plt.figure()
    plt.plot(epoch_losses, label="Training loss")
    plt.legend()
    plt.savefig("./reports/figures/loss.png")

@hydra.main(version_base=None, config_name="config.yaml", config_path="conf")
def main(cfg) -> None: # TODO: Add typing for hydra cfg 
    # Set up hyper-parameters # TODO: What to do with these?
    #device = "cuda" if torch.cuda.is_available() else "cpu"
    device = 'cpu'
    path = "data/processed/train.csv"

    # Load training data 
    train_set = load_dataset(path)
    
    train_params = {"batch_size": cfg.training.hyperparameters.train_batch_size, "shuffle": True, "num_workers": 0}
    train_loader = DataLoader(train_set, **train_params)

    # Initialize model, objective and optimizer 
    model = BERT(drop_p = cfg.model.hyperparameters.drop_p, embed_dim = cfg.model.hyperparameters.embed_dim, out_dim = cfg.model.hyperparameters.out_dim).to(device)
    criterion = BCEWithLogitsLoss()
    optimizer = Adam(params=model.parameters(), lr=cfg.training.hyperparameters.learning_rate)

    # Train model 
    train(cfg, model, criterion, optimizer, train_loader, device)

if __name__ == "__main__":
    main()
