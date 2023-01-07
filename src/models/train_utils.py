import torch


def loss_fn(outputs, targets):
    """
    Calculate the binary cross entropy loss with logits for a batch of outputs and targets.

    Args:
        outputs (torch.Tensor): Model outputs of shape (batch_size, num_classes).
        targets (torch.Tensor): Target values of shape (batch_size, num_classes).

    Returns:
        torch.Tensor: Binary cross entropy loss with logits for the given outputs and targets.
    """
    return torch.nn.BCEWithLogitsLoss()(outputs, targets)
