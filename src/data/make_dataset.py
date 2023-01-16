import logging
from pathlib import Path

import click
import pandas as pd
from dotenv import find_dotenv, load_dotenv


@click.command()
@click.argument("input_filepath", type=click.Path(exists=True))
@click.argument("output_filepath", type=click.Path())
def main(input_filepath: str, output_filepath: str) -> None:
    """
    Turns csv-files [../raw/*.csv] of raw data into 2 correctly formated
    dataframes of processed train and test data saved as a csv.file
    [../processed]

    :param input_filepath: Path to directory (e.g. './src/data/raw')
    :param output_filepath: Path to directory (e.g. './src/data/processed')
    """

    logger = logging.getLogger(__name__)
    logger.info("making final data set from raw data")

    # Load as dataframe
    df_sub = pd.read_csv(input_filepath + "/submit.csv")
    df_test = pd.read_csv(input_filepath + "/test.csv")
    df_train = pd.read_csv(input_filepath + "/train.csv")

    # Delete author column
    df_test = df_test.drop("author", axis=1)
    df_train = df_train.drop("author", axis=1)

    # Get test_labels and make train labels dataframe
    df_test_labels = df_sub[df_sub["id"].isin(df_test["id"])]

    df_train_labels = pd.DataFrame()
    df_train_labels["label"] = df_train["label"].tolist()
    df_train = df_train.drop("label", axis=1)

    # Drop id from all dataframes
    df_test = df_test.drop("id", axis=1)
    df_train = df_train.drop("id", axis=1)
    df_test_labels = df_test_labels.drop("id", axis=1)

    # One hot encode labels
    one_hot_test = pd.get_dummies(df_test_labels["label"])
    one_hot_train = pd.get_dummies(df_train_labels["label"])

    # Make one hot encoding into list (single column)
    test_labels = pd.DataFrame()
    train_labels = pd.DataFrame()

    test_labels["list"] = one_hot_test.values.tolist()
    train_labels["list"] = one_hot_train.values.tolist()

    # Merge title and text
    df_train["comment_text"] = df_train["title"] + " " + df_train["text"]
    df_test["comment_text"] = df_test["title"] + " " + df_test["text"]

    df_train = df_train.drop("title", axis=1)
    df_train = df_train.drop("text", axis=1)

    df_test = df_test.drop("title", axis=1)
    df_test = df_test.drop("text", axis=1)

    # Collect text and labels
    df_train["list"] = train_labels["list"]
    df_test["list"] = test_labels["list"]

    # Pickle the dataframe (to not loose list type)
    df_train.to_csv(output_filepath + "/train.csv", index=False)
    df_test.to_csv(output_filepath + "/test.csv", index=False)


if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
