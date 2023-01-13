import click
import hydra
import numpy as np
import torch
from sklearn import metrics
from torch.utils.data import DataLoader
from transformers import BertTokenizer
from tqdm import tqdm

from src.data.data_utils import load_dataset
from src.models.model import BERT


@click.command()
@click.argument("model_path")
@hydra.main(version_base=None, config_name="conf/config.yaml", config_path=".")
def test(cfg, model_path: str) -> None:  # TODO: Add typing for hydra cfg
    """
    Run model on the test set

    :param cfg: Config object with hyperparameters
    :param model_path: File path to the saved model weights
    """

    device = (
        "cuda" if torch.cuda.is_available() else "cpu"
    )  # TODO: Device should included
    path = "data/processed/test.csv"  # TODO: Add to config

    # Loading test set
    test_set = load_dataset(path)
    test_loader = DataLoader(test_set, batch_size=cfg.hps.valid_batch_size)

    # Initializing model
    model = BERT()
    model.load_state_dict(torch.load(model_path))

    # Test model
    model.eval()

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


if __name__ == "__main__":
    test()
