import hydra
import torch
from omegaconf import DictConfig
import click

from src.data.data_utils import load_txt_example
from src.models.model import BERT

@click.command()
@click.argument("path_weights", type=click.Path(exists=True))
@hydra.main(version_base=None, 
            config_name="config.yaml", 
            config_path="../../hydra_config")
def predict(cfg: DictConfig, path_weights:str) -> None:
    """ 
    Run prediction on a single txt-example 
    
    :param cfg: configuration file
    :return: prediction label
    """

    # Load data and tokenize it
    ids, mask, token_type_ids = load_txt_example(cfg, cfg.pred.path_data)

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

    ids = ids.to(device, dtype=torch.long)
    mask = mask.to(device, dtype=torch.long)
    token_type_ids = token_type_ids.to(device, dtype=torch.long)

    # Initialize model from weights
    model = BERT(drop_p=cfg.model.drop_p)

    # Load weights
    model.load_state_dict(torch.load(path_weights))
    model.to(device)

    # Run forward pass 
    model.eval()

    with torch.no_grad():
        outputs = model(ids, mask, token_type_ids)

    # Print results
    prediction = torch.max(outputs, 1)
    prediction = "unreliable" if prediction.indices[0] == 1 else "reliable"
    print(f"Prediction: {prediction}")


if __name__ == "__main__":
    predict()
    
