import os
from http import HTTPStatus

import torch
from fastapi import FastAPI, File, UploadFile
from google.cloud import storage

from src.data.data_utils import load_txt_example
from src.models.model import BERT

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "./application_default_credentials.json"

app = FastAPI()


class config:
    def __init__(self):
        self.model = self.model_hps()
        self.train = self.train_hps()
        self.pred = self.pred_hps()

    class model_hps:
        def __init__(self):
            self.device = "cpu"
            self.max_len = 20
            self.drop_p = 0.3
            self.embed_dim = 768
            self.out_dim = 2
            self.bert_version = "bert-base-uncased"

    class train_hps:
        def __init__(self) -> None:
            self.device = "cpu"
            self.max_len = 20
            self.drop_p = 0.3
            self.embed_dim = 768
            self.out_dim = 2
            self.bert_version = "bert-base-uncased"

    class pred_hps:
        def __init__(self) -> None:
            self.device = "cpu"
            self.max_len = 20
            self.drop_p = 0.3
            self.embed_dim = 768
            self.out_dim = 2
            self.bert_version = "bert-base-uncased"


@app.post("/")
async def read_root(data: UploadFile = File(...)):
    """
    Takes a txt-file as input and makes prediction with
    model saved in Google cloud bucket

    :param data: txt-file
    :return: response dictionary
    """

    # If file is a txt-file then read it
    if data.content_type == "text/plain":
        text_binary = await data.read()
        text_example = text_binary.decode("utf-8")

    if not os.path.exists("./temp_files"):
        os.mkdir("./temp_files")

    text_path = "./temp_files/temp.txt"
    weights_path = "./temp_files/weights.pt"

    # Save the text example to a temporary file
    with open(text_path, "w") as f:
        f.write(text_example)

    # Download weights from Google cloud bucket
    if not os.path.exists(weights_path):
        storage_client = storage.Client("dtu-mlops")
        bucket = storage_client.get_bucket("news_model_weights")
        blob = bucket.blob("weights.pt")
        blob.download_to_filename(weights_path)

    # Make config object
    cfg = config()

    # Load data and tokenize it
    ids, mask, token_type_ids = load_txt_example(cfg, text_path)

    device = cfg.pred.device
    ids = ids.to(device, dtype=torch.long)
    mask = mask.to(device, dtype=torch.long)
    token_type_ids = token_type_ids.to(device, dtype=torch.long)

    # Initialize model from weights
    model = BERT(drop_p=cfg.model.drop_p)

    # Load weights
    model.load_state_dict(torch.load(weights_path, map_location=torch.device("cpu")))
    model.to(cfg.pred.device)

    # Run forward pass
    model.eval()

    with torch.no_grad():
        outputs = model(ids, mask, token_type_ids)

    # Map prediction to label
    prediction = torch.max(outputs, 1)
    prediction = "unreliable" if prediction.indices[0] == 1 else "reliable"

    # Return response
    response = {
        "text_example": text_example,
        "prediction": prediction,
        "HTTP status": HTTPStatus.OK.phrase,
        "HTTP status code": HTTPStatus.OK.value,
    }

    return response
