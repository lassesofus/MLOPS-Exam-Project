import torch
import transformers


class BERT(torch.nn.Module):  # TODO: Read more about BERT input/output
    def __init__(self, drop_p: float, embed_dim: int, out_dim: int) -> None:
        """
        Initialize a BERT model from pre-trained weights with an additional
        linear layer for fine-tuning.

        :param drop_p: The dropout probability
        :param embed_dim: The output dimension of the BERT model
        :param out_dim: The output dimension of the linear layer (2 classes)
        """
        super(BERT, self).__init__()
        # Constant parameters
        self.drop_p = drop_p
        self.embed_dim = embed_dim  # base-bert
        self.out_dim = out_dim  # binary classification
        # Initializing the BERT model from the "bert-base-uncased"
        # pre-trained model
        self.bert = transformers.BertModel.from_pretrained("bert-base-uncased")

        # Initializing dropout layer
        self.dropout = torch.nn.Dropout(self.drop_p)
        # Initializing linear layer
        self.linear = torch.nn.Linear(self.embed_dim, self.out_dim)

    def forward(self, x: torch.Tensor, batch_size: int) -> torch.Tensor:
        """
        Forward pass of the model (without loss calculation)

        :param x: bert model output
        :param batch_size: number of elements in one batch
        :returns: Model output of shape (batch_size, num_classes)
        """

        if x.ndim != 2:
            raise ValueError("Expected input to be a 2D tensor")
        if x.shape[0] != batch_size or x.shape[1] != self.embed_dim:
            raise ValueError("Wrong shape of the input")
        # Applying dropout
        x = self.dropout(x)
        if x.ndim != 2:
            raise ValueError("Expected input to be a 2D tensor")
        if x.shape[0] != batch_size or x.shape[1] != self.embed_dim:
            raise ValueError("Wrong shape")

        # Applying linear layer
        x = self.linear(x)
        if x.ndim != 2:
            raise ValueError("Expected input to be a 2D tensor")
        if x.shape[0] != batch_size or x.shape[1] != self.out_dim:
            raise ValueError("Wrong shape")

        return x
