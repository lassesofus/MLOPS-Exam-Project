import hydra
import torch
from torch.utils.data import DataLoader
from transformers import BertTokenizer

from src.data.dataset_class import get_dataset
from src.models.models import BERTClass
from src.models.train_utils import loss_fn

# The train function uses the Hydra library to handle command line arguments and configuration files.
# The function takes in a single argument, `cfg`, which is a Hydra Conf object that contains the configuration 
# for the training process.
@hydra.main(version_base=None, config_name="config.yaml", config_path=".")
def train(cfg):

    """
    Train a BERT model on the specified dataset.

    Args:
        cfg (hydra.core.config.CompositeConf): Configuration object containing the hyperparameters
            for training and the paths to the dataset.
    """

    # Setting up the device for GPU usage if available, otherwise using CPU
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Extracting key variables from the config for use in training
    TRAIN_BATCH_SIZE = cfg.hyperparameters.train_batch_size
    EPOCHS = cfg.hyperparameters.epochs
    LEARNING_RATE = cfg.hyperparameters.learning_rate


    # Defining the parameters for the training DataLoader
    train_params = {"batch_size": TRAIN_BATCH_SIZE, 
                    "shuffle": True, 
                    "num_workers": 0}

    # Loading the training set
    path = "data/processed/train.csv"
    train_set = get_dataset(path)
    training_loader = DataLoader(train_set, **train_params)

    # Initializing the model and moving it to the designated device
    model = BERTClass()
    model.to(device)

    # Initializing the optimizer
    optimizer = torch.optim.Adam(params=model.parameters(), lr=LEARNING_RATE)


    # Training loop
    for epoch in range(EPOCHS):

        # Set model to training mode
        model.train()

        # Iterate over training data
        for _, data in enumerate(training_loader, 0):
            # Move data to device
            ids = data["ids"].to(device, dtype=torch.long)
            mask = data["mask"].to(device, dtype=torch.long)
            token_type_ids = data["token_type_ids"].to(device, dtype=torch.long)
            targets = data["targets"].to(device, dtype=torch.float)

            # Get model outputs
            outputs = model(ids, mask, token_type_ids)

            # Calculate loss and print every 5000 iterations
            loss = loss_fn(outputs, targets)
            if _ % 5000 == 0:
                print(f"Epoch: {epoch}, Loss:  {loss.item()}")

            # Backpropagate and update weights
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    # save the model
    torch.save(model.state_dict(), "trained_model.pt")


if __name__ == "__main__":
    train()
