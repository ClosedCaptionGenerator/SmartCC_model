import torch
import numpy as np


def one_hot_encode(labels, num_classes):
    return np.eye(num_classes)[labels]


def accuracy(predictions, targets):
    _, preds = torch.max(predictions, dim=1)
    return torch.sum(preds == targets).item() / len(targets)