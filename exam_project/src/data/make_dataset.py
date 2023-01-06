# -*- coding: utf-8 -*-
import click
import logging
from pathlib import Path
from dotenv import find_dotenv, load_dotenv

# TODO: Remove unecessary modules
import numpy as np
import pandas as pd
#from sklearn import metrics
#import transformers
#import torch
#from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler
from transformers import BertTokenizer

#@click.command()
#@click.argument('input_filepath', type=click.Path(exists=True))
#@click.argument('output_filepath', type=click.Path())
def main(input_filepath=None, output_filepath=None):
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info('making final data set from raw data')

    """ Part 1: Turn CSV into dataframe (comment_text, list)"""
    # Load as dataframe 
    df_sub = pd.read_csv("./exam_project/data/raw/submit.csv")
    df_test = pd.read_csv("./exam_project/data/raw/test.csv")
    df_train = pd.read_csv("./exam_project/data/raw/train.csv")

    # Delete author column
    df_test = df_test.drop('author', axis=1)
    df_train = df_train.drop('author', axis=1)

    # Split labels intro train and test dataframe 
    df_train_labels = df_sub[df_sub['id'].isin(df_train['id'])] # Debug empty
    df_test_labels = df_sub[df_sub['id'].isin(df_test['id'])]

    # Drop id from all dataframes 
    df_test = df_test.drop('id', axis=1)
    df_train = df_train.drop('id', axis=1)

    df_test_labels = df_test_labels.drop('id', axis=1)
    df_train_labels = df_train_labels.drop('id', axis=1)

    # One hot encode labels 
    one_hot_test = pd.get_dummies(df_test_labels['label'])
    one_hot_train = pd.get_dummies(df_train_labels['label'])

    df_test_labels = df_test_labels.drop('label',axis = 1)
    df_train_labels = df_train_labels.drop('label',axis = 1)
    
    df_test_labels = df_test_labels.join(one_hot_test)
    df_train_labels = df_train_labels.join(one_hot_train)

    # Make one hot encoding into list (single column)
    df_test_labels['list'] = df_test_labels.values.tolist() 
    df_train_labels['list'] = df_train_labels.values.tolist() 

    df_test_labels = df_test_labels.drop(1,axis = 1)
    df_test_labels = df_test_labels.drop(0,axis = 1)

    #df_train_labels = df_train_labels.drop('1',axis = 1)
    #df_train_labels = df_train_labels.drop('0',axis = 1)

    # 


    """ Part 2: Defining key variables """
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    MAX_LEN = 200

    """ Part 3: Change to BERT format """
    def bert_format():
        comment_text = str(comment_text[index])
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














if __name__ == '__main__':
    #log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    #logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    #project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    #load_dotenv(find_dotenv())

    main()
