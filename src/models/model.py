import torch
import transformers

class BERT(torch.nn.Module): # TODO: Read more about BERT input/output 
    def __init__(self, drop_p: float) -> None:
        """ 
        Initialize a BERT model from pre-trained weights with an additional 
        linear layer for fine-tuning. 

        :param drop_p: The dropout probability 
        :param embed_dim: The output dimension of the BERT model 
        :param out_dim: The output dimension of the linear layer (2 classes) 
        """
        super(BERT, self).__init__()
        
        # Constant parameters 
        embed_dim = 768 # base-bert 
        out_dim: 2 # binary classification 

        # Initializing the BERT model from the "bert-base-uncased" pre-trained model
        self.bert = transformers.BertModel.from_pretrained("bert-base-uncased")
        
        # Initializing dropout layer
        self.dropout = torch.nn.Dropout(drop_p)

        # Initializing linear layer 
        self.linear = torch.nn.Linear(embed_dim, out_dim)

    def forward(self, ids:torch.Tensor, mask:torch.Tensor, 
                token_type_ids:torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the model (without loss calculation)

        :param ids: Input ids of shape (batch_size, sequence_length)
        :param mask: Attention mask of shape (batch_size, sequence_length)
        :param token_type_ids: Token type ids of shape (batch_size, sequence_length)
        :returns: Model output of shape (batch_size, num_classes)
        """

        # Getting the BERT model output and ignoring the pooled output # TODO: Check what is pooled output
        _, x = self.bert(ids, attention_mask=mask, token_type_ids=token_type_ids, return_dict=False)

        # Applying dropout and linear layer
        x = self.dropout(x)
        x = self.linear(x)

        return x