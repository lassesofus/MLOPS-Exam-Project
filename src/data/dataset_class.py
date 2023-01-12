import ast

import hydra
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import BertTokenizer


class CustomDataset(Dataset):
    """Custom dataset for loading Kaggle Toxic Comment Classification data

    Classes:
        Dataset from torch.utils.data

    """

    def __init__(self, dataframe, tokenizer, max_len):
        """
        Args:
            dataframe: Pandas dataframe containing the data
            tokenizer: Tokenizer to use for encoding comments
            max_len: Maximum length of encoded comments
        """
        self.tokenizer = tokenizer
        self.data = dataframe
        self.comment_text = dataframe.comment_text
        self.targets = self.data.list
        self.max_len = max_len

    def __len__(self):
        """Returns the length of the data"""
        return len(self.comment_text)

    def __getitem__(self, index):
        """
        Returns the encoded comments and target labels for a given index

        Args:
            index: Index of the data to return

        Returns:
            A dictionary containing the following elements:
                ids: Encoded comment tensor
                mask: Attention mask tensor
                token_type_ids: Token type ids tensor
                targets: Target label tensor
        """
        comment_text = str(self.comment_text[index])
        comment_text = " ".join(comment_text.split())

        inputs = self.tokenizer.encode_plus(
            comment_text,
            None,
            add_special_tokens=True,
            max_length=self.max_len,
            pad_to_max_length=True,
            return_token_type_ids=True,
        )
        ids = inputs["input_ids"]
        mask = inputs["attention_mask"]
        token_type_ids = inputs["token_type_ids"]

        return {
            "ids": torch.tensor(ids, dtype=torch.long),
            "mask": torch.tensor(mask, dtype=torch.long),
            "token_type_ids": torch.tensor(token_type_ids, dtype=torch.long),
            "targets": torch.tensor(self.targets[index], dtype=torch.float),
        }


def get_dataset(path_file):
    """
    Loads and returns the Kaggle Toxic Comment Classification dataset

    Args:
        path_file: Path to the CSV file containing the dataset

    Returns:
        A CustomDataset object
    """

    df = pd.read_csv(path_file)
    df["list"] = df["list"].apply(lambda x: list(map(int, x.strip("][").split(", "))))
    df = df.reset_index(drop=True)

    print(f"Dataset: {df.shape}")

    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    max_len = 200
    dataset = CustomDataset(df, tokenizer, max_len)
    print(list(dataset.__dict__.keys())[0])
    return dataset


get_dataset("data/processed/train.csv")
