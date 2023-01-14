import os

import pytest
from src.data.data_utils import load_dataset
from tests import _PATH_DATA


@pytest.mark.skipif(not os.path.exists(_PATH_DATA),
                    reason="Data files not found")
def test_numb_obs() -> None:
    # Check that the lengths of the datasets are as expected
    # Paths for data
    path_train = "data/processed/train.csv"
    path_test = "data/processed/test.csv"

    # Load train and test datasets
    train_set = load_dataset(path_train)
    test_set = load_dataset(path_test)
    assert (
        len(train_set) == 20800 and len(test_set) == 5200
    ), "Dataset splits did not have expected lengths"


def test_numb_keys():
    # Check that the datasets have the expected number of dictionary keys
    # Paths for data
    path_train = "data/processed/train.csv"

    # Load train and test datasets
    train_set = load_dataset(path_train)
    assert (
        len(train_set.__dict__.keys()) == 5
    ), "Max length is not expected value"


def test_dict_key1() -> None:
    # Check that the datasets have the expected key
    # Paths for data
    path_train = "data/processed/train.csv"

    # Load train and test datasets
    train_set = load_dataset(path_train)
    assert (
        list(train_set.__dict__.keys())[0] == "tokenizer"
    ), "The first key is not 'tokenizer'"


def test_dict_key2() -> None:
    # Check that the datasets have the expected key
    # Paths for data
    path_train = "data/processed/train.csv"

    # Load train and test datasets
    train_set = load_dataset(path_train)
    assert (
        list(train_set.__dict__.keys())[1] == "data"
    ), "The second key is not 'data'"


def test_dict_key3() -> None:
    # Check that the datasets have the expected key
    # Paths for data
    path_train = "data/processed/train.csv"

    # Load train and test datasets
    train_set = load_dataset(path_train)
    assert (
        list(train_set.__dict__.keys())[2] == "comment_text"
    ), "The third key is not 'comment_text'"


def test_dict_key4() -> None:
    # Check that the datasets have the expected key
    # Paths for data
    path_train = "data/processed/train.csv"

    # Load train and test datasets
    train_set = load_dataset(path_train)
    assert (
        list(train_set.__dict__.keys())[3] == "targets"
    ), "The fourth key is not 'targets'"


def test_dict_key5() -> None:
    # Paths for data
    path_train = "data/processed/train.csv"

    # Load train and test datasets
    train_set = load_dataset(path_train)
    # Check that the datasets have the expected key
    assert (
        list(train_set.__dict__.keys())[4] == "max_len"
    ), "The fifth key is not 'max_len'"
