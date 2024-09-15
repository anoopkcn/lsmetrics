import torch.nn.functional as F


def mean_absolute_error(pred, target):
    return F.l1_loss(pred, target)
