import click
import hydra
import torch
from omegaconf import DictConfig

from src.data.data_utils import load_txt_example
from src.models.model import BERT

@hydra.main(version_base=None, config_name="config.yaml", config_path="conf")
def main(cfg: DictConfig) -> None:

    model_path = cfg.predicting.paths.model_path
    data_path = cfg.predicting.paths.data_path
    bert_version = cfg.model.hyperparameters.bert_version  
    max_len = cfg.model.hyperparameters.max_len  

    # Load data and tokenize it
    ids, mask, token_type_ids = load_txt_example(data_path, max_len,
                                                 bert_version)

    # Initialize model from weights
    model = BERT(drop_p=cfg.model.hyperparameters.drop_p,
                embed_dim=cfg.model.hyperparameters.embed_dim,
                out_dim=cfg.model.hyperparameters.out_dim)
    model.load_state_dict(torch.load(model_path))

    # Run model forward pass
    model.eval()
    with torch.no_grad():
        outputs = model(ids, mask, token_type_ids)

    # Print results
    prediction = torch.max(outputs, 1)
    prediction = "unreliable" if prediction.indices[0] == 1 else "reliable"
    print(f"Prediction: {prediction}")


if __name__ == "__main__":
    main()
