import click
import hydra
import torch

from src.data.data_utils import load_txt_example
from src.models.model import BERT


@click.command()
@click.argument("model_path")
@click.argumnet("data_path")
@hydra.main(version_base=None, config_name="conf/config.yaml", config_path=".")
def main(model_path: str, data_path: str) -> None:
    # Add to hyperparameters
    bert_version = "bert-base-uncased"  # TODO: Turn into hyperparameter
    max_len = 20  # TODO: Turn into hyperparameter

    # Load data and tokenize it
    ids, mask, token_type_ids = load_txt_example(data_path, bert_version,
                                                 max_len)

    # Initialize model from weights
    model = BERT()
    model.load_state_dict(torch.load(model_path))

    # Run model forward pass
    model.eval()
    with torch.no_grad():
        outputs = model(ids, mask, token_type_ids)

    # Print results
    prediction = torch.max(outputs, 1).tolist()
    print(f"Prediction: {prediction}")


if __name__ == "__main__":
    main()
