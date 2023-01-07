import hydra
import torch 
from transformers import BertTokenizer
from torch.utils.data import DataLoader
import click
from sklearn import metrics
import numpy as np

from src.data.dataset_class import get_dataset
from src.models.models import BERTClass
from src.models.train_utils import loss_fn

# The train function uses the Hydra library to handle command line arguments and configuration files.
# The function takes in a single argument, `cfg`, which is a Hydra Conf object that contains the configuration 
# for the training process.
@hydra.main(version_base=None, config_name="config.yaml", config_path=".")
@click.command()
@click.argument("model_path")
def test(model_path,cfg):

    """
    The main function for testing the trained model.

    Parameters:
        model_path (str): The file path to the saved model.
        cfg (Conf): The Hydra Conf object containing the configuration for the training process.
    
    Returns:
        None
    """

    # Setting up the device for GPU usage if available, otherwise using CPU
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Extracting key variables from the config for use in testing
    VALID_BATCH_SIZE = cfg.hyperparameters.valid_batch_size
    EPOCHS = cfg.hyperparameters.epochs

    # Defining the parameters for the training DataLoader
    test_params = {'batch_size': VALID_BATCH_SIZE,
                'shuffle': True,
                'num_workers': 0
                }

    # Loading the training set
    path = "data/processed/train.csv"
    train_set = get_dataset(path)
    testing_loader = DataLoader(train_set, **test_params)

    # Initializing the model and moving it to the designated device
    model = BERTClass()
    model.load_state_dict(torch.load(model_path))

    # Training loop
    for epoch in range(EPOCHS):

        model.eval()
        fin_targets=[]
        fin_outputs=[]
        with torch.no_grad():
            for _, data in enumerate(testing_loader, 0):
                # Extracting data from the data batch
                ids = data['ids'].to(device, dtype = torch.long)
                mask = data['mask'].to(device, dtype = torch.long)
                token_type_ids = data['token_type_ids'].to(device, dtype = torch.long)
                targets = data['targets'].to(device, dtype = torch.float)
                # Running the model on the data to get the outputs
                outputs = model(ids, mask, token_type_ids)
                # Appending the targets and outputs to lists
                fin_targets.extend(targets.cpu().detach().numpy().tolist())
                fin_outputs.extend(torch.sigmoid(outputs).cpu().detach().numpy().tolist())

        # Converting the output and target lists to outputs and targets
        outputs = fin_outputs
        targets = fin_targets
        outputs = np.array(outputs) >= 0.5
        accuracy = metrics.accuracy_score(targets, outputs)
        f1_score_micro = metrics.f1_score(targets, outputs, average='micro')
        f1_score_macro = metrics.f1_score(targets, outputs, average='macro')
        print(f"Accuracy Score = {accuracy}")
        print(f"F1 Score (Micro) = {f1_score_micro}")
        print(f"F1 Score (Macro) = {f1_score_macro}")


if __name__ == "__main__":
    test()