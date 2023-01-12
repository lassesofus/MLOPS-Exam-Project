# -*- coding: utf-8 -*-
#!/usr/bin/python
import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import BertTokenizer
from typing import List, Dict 


class fake_news_dataset(Dataset):
    def __init__(self, dataframe:pd.DataFrame, tokenizer:BertTokenizer, 
                max_len:int) -> None:
        """ 
        Initialize fake news dataset class 

        :param dataframe: Pandas dataframe containing processed data
        :param tokenizer: Tokenizer to encode text to BERT input 
        :param max_len: Maximum length of encoded text (crop if above)
        """

        self.tokenizer = tokenizer
        self.data = dataframe
        self.comment_text = dataframe.comment_text
        self.targets = self.data.list
        self.max_len = max_len

    def __len__(self) -> int:
        """ 
        :returns: Amount of observations in dataset 
        """
        return len(self.comment_text)

    def __getitem__(self, index:List[int]): # TODO: Don't know if this is correct typing 

        #Dict[torch.Tensor]
        """ Returns the encoded comments and target labels for a given index

        :param index: Dataframe index of the data to return
        :returns: A dictionary containing the following elements:
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

def load_dataset(path_file:str) -> Dataset: # TODO: Typing correct - or use fake_news_dataset?
    """
    Load dataset from CSV file with one-hot-encoded labels in correct format (list of ints)

    :param path_file: Path to the CSV file containing the processed dataset
    :returns: A fake_news_dataset object
    """

    # Load dataframe with correct format lists
    df = pd.read_csv(path_file)
    df["list"] = df["list"].apply(lambda x: list(map(int, x.strip("][").split(", "))))
    df = df.reset_index(drop=True)

    # Load tokenizer
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    # Define max len # TODO: Make into hyperparameter using Hydra 
    max_len = 20

    # Initialize dataset object
    dataset = fake_news_dataset(df, tokenizer, max_len)

    return dataset


if __name__=="__main__":
    load_dataset("data/processed/train.csv")


