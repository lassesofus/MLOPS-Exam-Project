from transformers import BertTokenizer
from torch import cuda
from torch.utils.data import Dataset, DataLoader
import torch 

from exam_project.models.models import BERTClass
from exam_project.src.models.train_utils import loss_fn 
import hydra

@hydra.main(version_base=None, config_name="config.yaml", config_path=".")

def train(cfg):
    # # Setting up the device for GPU usage
    device = 'cuda' if cuda.is_available() else 'cpu'

    # Sections of config

    # Defining some key variables that will be used later on in the training
    MAX_LEN = cfg.hyperparameters.batch_size.max_len
    TRAIN_BATCH_SIZE = cfg.hyperparameters.batch_size.train_batch_size
    VALID_BATCH_SIZE = cfg.hyperparameters.batch_size.valid_batch_size
    EPOCHS = cfg.hyperparameters.batch_size.epochs
    LEARNING_RATE = cfg.hyperparameters.batch_size.learning_rate
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    train_params = {'batch_size': TRAIN_BATCH_SIZE,
                'shuffle': True,
                'num_workers': 0
                }

    model = BERTClass()
    model.to(device)

    optimizer = torch.optim.Adam(params =  model.parameters(), lr=LEARNING_RATE)

    train_set = torch.load("data/processed/train_set.pt")
    training_loader = DataLoader(train_set, **train_params)

    for epoch in range(EPOCHS):

        model.train()
        for _,data in enumerate(training_loader, 0):
            ids = data['ids'].to(device, dtype = torch.long)
            mask = data['mask'].to(device, dtype = torch.long)
            token_type_ids = data['token_type_ids'].to(device, dtype = torch.long)
            targets = data['targets'].to(device, dtype = torch.float)

            outputs = model(ids, mask, token_type_ids)

            optimizer.zero_grad()
            loss = loss_fn(outputs, targets)
            if _%5000==0:
                print(f'Epoch: {epoch}, Loss:  {loss.item()}')
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    torch.save(model.state_dict(), 'trained_model.pt')


if __name__ == "__main__":
   train()
