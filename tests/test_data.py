import os
import pytest
from src.data.make_dataset import load_dataset


@pytest.mark.skipif(not os.path.exists("./data"), reason="Data files not found")
class TestData:  # TODO: Add more tests and docstrings
    def __init__(self) -> None:
        super().__init__()  # TODO: Does it inherent correctly now

        # Paths for data
        path_train = "data/processed/train.csv"
        path_test = "data/processed/test.csv"

        # Load train and test datasets
        self.train_set = load_dataset(path_train)
        self.test_set = load_dataset(path_test)

    def test_numb_obs(self) -> None:
        # Check that the lengths of the datasets are as expected
        assert (
            len(self.train_set) == 20800 and len(self.test_set) == 5200
        ), "Dataset splits did not have expected lengths"

    def test_numb_keys(self):
        # Check that the datasets have the expected number of dictionary keys
        assert (
            len(self.train_set.__dict__.keys()) == 5
        ), "Max length is not expected value"

    def test_dict_key1(self) -> None:
        # Check that the datasets have the expected key
        assert (
            list(self.train_set.__dict__.keys())[0] == "tokenizer"
        ), "The first key is not 'tokenizer'"

    def test_dict_key2(self) -> None:
        # Check that the datasets have the expected key
        assert (
            list(self.train_set.__dict__.keys())[1] == "data"
        ), "The second key is not 'data'"

    def test_dict_key3(self) -> None:
        # Check that the datasets have the expected key
        assert (
            list(self.train_set.__dict__.keys())[2] == "comment_text"
        ), "The third key is not 'comment_text'"

    def test_dict_key4(self) -> None:
        # Check that the datasets have the expected key
        assert (
            list(self.train_set.__dict__.keys())[3] == "targets"
        ), "The fourth key is not 'targets'"

    def test_dict_key5(self) -> None:
        # Check that the datasets have the expected key
        assert (
            list(self.train_set.__dict__.keys())[4] == "max_len"
        ), "The fifth key is not 'max_len'"
