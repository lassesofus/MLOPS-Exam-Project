import torch
import transformers

# Added comment


class BERTClass(torch.nn.Module):
    def __init__(self, drop_p: float = 0.5, hidden_layers: int = 768, output_size: int = 2):
        """
        Initialize a BERT model for classification.

        Arguments
        ---------
        input_size: integer, size of the input layer
        output_size: integer, size of the output layer

        Classes: 
            torch.nn.Module

        Functions: 
            forward(ids,mask,token_type_ids) -> torch.tensor
        """
        super(BERTClass, self).__init__()
        # Initializing the BERT model from the "bert-base-uncased" pre-trained model
        self.l1 = transformers.BertModel.from_pretrained("bert-base-uncased")
        # Initializing a dropout layer with a dropout rate of 0.3
        self.l2 = torch.nn.Dropout(drop_p)
        # Initializing a linear layer with output dimension 2
        self.l3 = torch.nn.Linear(hidden_layers, output_size)

    def forward(self, ids, mask, token_type_ids):
        """
        Forward pass of the model.

        Args:
            ids (torch.Tensor): Input ids of shape (batch_size, sequence_length).
            mask (torch.Tensor): Attention mask of shape (batch_size, sequence_length).
            token_type_ids (torch.Tensor): Token type ids of shape (batch_size, sequence_length).

        Returns:
            torch.Tensor: Model output of shape (batch_size, num_classes).
        """
        
        # Getting the BERT model output and ignoring the pooled output
        _, output_1 = self.l1(ids, attention_mask=mask, token_type_ids=token_type_ids,return_dict=False)
        output_2 = self.l2(output_1)
        output = self.l3(output_2)
        
        return output
