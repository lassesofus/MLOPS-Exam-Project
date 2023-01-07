import hydra
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizer

from src.data.dataset_class import get_dataset
from src.models.models import BERTClass
from src.models.train_utils import loss_fn


@hydra.main(version_base=None, config_name="config.yaml", config_path=".")
def train(cfg):
    # # Setting up the device for GPU usage
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Sections of config

    # Defining some key variables that will be used later on in the training
    MAX_LEN = cfg.hyperparameters.max_len
    TRAIN_BATCH_SIZE = cfg.hyperparameters.train_batch_size
    VALID_BATCH_SIZE = cfg.hyperparameters.valid_batch_size
    EPOCHS = cfg.hyperparameters.epochs
    LEARNING_RATE = cfg.hyperparameters.learning_rate

    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    train_params = {"batch_size": TRAIN_BATCH_SIZE, "shuffle": True, "num_workers": 0}

    model = BERTClass()
    model.to(device)

    optimizer = torch.optim.Adam(params=model.parameters(), lr=LEARNING_RATE)

    path = "data/processed/train.csv"
    train_set = get_dataset(path)
    training_loader = DataLoader(train_set, **train_params)

    for epoch in range(EPOCHS):

        model.train()
        for _, data in enumerate(training_loader, 0):
            ids = data["ids"].to(device, dtype=torch.long)
            mask = data["mask"].to(device, dtype=torch.long)
            token_type_ids = data["token_type_ids"].to(device, dtype=torch.long)
            targets = data["targets"].to(device, dtype=torch.float)

            outputs = model(ids, mask, token_type_ids)

            optimizer.zero_grad()
            loss = loss_fn(outputs, targets)
            if _ % 5000 == 0:
                print(f"Epoch: {epoch}, Loss:  {loss.item()}")

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    torch.save(model.state_dict(), "trained_model.pt")


if __name__ == "__main__":
    train()
