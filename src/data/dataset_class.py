import numpy as np
import pandas as pd
from sklearn import metrics
import torch
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler
from transformers import BertTokenizer, BertModel, BertConfig
import ast

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
            return_token_type_ids=True
        )
        ids = inputs['input_ids']
        mask = inputs['attention_mask']
        token_type_ids = inputs["token_type_ids"]

        return {
            'ids': torch.tensor(ids, dtype=torch.long),
            'mask': torch.tensor(mask, dtype=torch.long),
            'token_type_ids': torch.tensor(token_type_ids, dtype=torch.long),
            'targets': torch.tensor(self.targets[index], dtype=torch.float)
        }

if __name__ == "__main__":
    # For debugging purpose 
    df_train = pd.read_csv('./exam_project/data/processed/train.csv')
    df_train['list'] = df_train['list'].apply(lambda x: list(map(int,x.strip('][').split(', '))))

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    MAX_LEN = 200

    lol = CustomDataset(df_train, tokenizer, MAX_LEN)
    lol2 = lol.__getitem__(index=[0,1,2])

    #print('RUN MAIN FOR DEBUGGING')




