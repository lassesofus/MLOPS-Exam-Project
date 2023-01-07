import ast

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import BertTokenizer


class CustomDataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_len):
        self.tokenizer = tokenizer
        self.data = dataframe
        self.comment_text = dataframe.comment_text
        self.targets = self.data.list
        self.max_len = max_len

    def __len__(self):
        return len(self.comment_text)

    def __getitem__(self, index):
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
    df = pd.read_csv(path_file)
    df["list"] = df["list"].apply(lambda x: list(map(int, x.strip("][").split(", "))))
    df = df.reset_index(drop=True)

    print(f"Dataset: {df.shape}")

    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    MAX_LEN = 200

    dataset = CustomDataset(df, tokenizer, MAX_LEN)
    return dataset


# dataset = get_dataset("data/processed/train.csv")

# print(dataset)
