import os
#import hydra
import pytest
import torch
from tests import _PATH_DATA
from src.data.dataset_class import get_dataset





#@hydra.main(version_base=None, config_name="config.yaml", config_path=".")
@pytest.mark.skipif(not os.path.exists('data/processed/train.csv'), reason="Data files not found")
def test_data():
    """
    Test function to ensure that the train and test datasets have the correct shape and keys.

    The shape of the training set should have a shape of (20800, 5) and the test set should have a shape of (5200, 5).
    Each dataset should have the following dictionary elements:
        - ids: Encoded comment tensor
        - mask: Attention mask tensor
        - token_type_ids: Token type ids tensor
        - targets: Target label tensor
    """
    # N_train = cfg.dataset.N_train
    # N_test = cfg.dataset.N_test
    # columns = cfg.dataset.columns
    
    #Define the expected values 
    #Should be implemented with hydra!!!
    N_train = 20800
    N_test = 5200
    columns = 5

    #Define paths for data
    path_train = 'data/processed/train.csv'
    path_test = 'data/processed/test.csv'

    # Load train and test datasets
    train_set = get_dataset(path_train)
    test_set = get_dataset(path_test)
        
    # Check that the lengths of the datasets are as expected
    assert len(train_set) == N_train and len(test_set) == N_test , "Dataset splits did not have expected lengths"
    
    # Check that the datasets have the expected number of dictionary keys
    assert len(train_set.__dict__.keys()) == columns, "Max length is not expected value"
    
    # Check that the datasets have the expected keys
    assert list(train_set.__dict__.keys())[0] == "tokenizer", "The first key is not 'tokenizer'"
    assert list(train_set.__dict__.keys())[1] == "data", "The second key is not 'data'"
    assert list(train_set.__dict__.keys())[2] == "comment_text", "The third key is not 'comment_text'"
    assert list(train_set.__dict__.keys())[3] == "targets", "The fourth key is not 'targets'"
    assert list(train_set.__dict__.keys())[4] == "max_len", "The fifth key is not 'max_len'"

if __name__ == "__main__":
    test_data()     