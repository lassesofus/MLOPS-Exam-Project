import os

import pytest
import torch
import transformers
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.data.data_utils import load_dataset
from src.models.model import BERT


@pytest.mark.skipif(not os.path.exists("./data"),
                    reason="Data files not found")
def test_model_output_dimension():
    # Paths for data
    path_train = "data/processed/train.csv"

    # Load train and test datasets
    train_set = load_dataset(path_train)
    batch_size = 8
    train_loader = DataLoader(train_set, batch_size, shuffle=True)
    epochs = 1
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = BERT(drop_p=0.5, embed_dim=768, out_dim=2).to(device)

    with tqdm(train_loader, desc=f"Epoch {epochs}") as tepoch:
        for _, data in enumerate(tepoch):

            # Move data to device
            ids = data["ids"].to(device, dtype=torch.long)
            mask = data["mask"].to(device, dtype=torch.long)
            token_type_ids = data["token_type_ids"].to(device, dtype=torch.long)
            targets = data["targets"].to(device, dtype=torch.float)
            # Getting the BERT model output and ignoring the pooled output
            # TODO: Check what is pooled output
            bert = transformers.BertModel.from_pretrained("bert-base-uncased")
            _, x = bert(
                ids,
                attention_mask=mask,
                token_type_ids=token_type_ids,
                return_dict=False,
            )
            # Forward pass and loss calculation
            outputs = model(x, batch_size)
            break
    assert outputs.size() == targets.size()


@pytest.mark.skipif(not os.path.exists("./data"),
                    reason="Data files not found")
def test_error_on_wrong_input_dimensions():
    drop_p = 0.5
    embed_dim = 768
    out_dim = 2
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = BERT(drop_p, embed_dim, out_dim).to(device)
    with pytest.raises(ValueError, match="Expected input to be a 2D tensor"):
        model(torch.randn(4), batch_size=8)


@pytest.mark.skipif(not os.path.exists("./data"),
                    reason="Data files not found")
def test_error_on_wrong_first_dimension():
    drop_p = 0.5
    embed_dim = 768
    out_dim = 2
    batch_size = 8
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = BERT(drop_p, embed_dim, out_dim).to(device)
    with pytest.raises(ValueError, match="Wrong shape"):
        model(torch.randn(batch_size + 1, embed_dim), batch_size)


@pytest.mark.skipif(not os.path.exists("./data"),
                    reason="Data files not found")
def test_error_on_wrong_second_dimension():
    drop_p = 0.5
    embed_dim = 768
    out_dim = 2
    batch_size = 8
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = BERT(drop_p, embed_dim, out_dim).to(device)
    with pytest.raises(ValueError, match="Wrong shape"):
        model(torch.randn(batch_size, embed_dim + 1), batch_size)
