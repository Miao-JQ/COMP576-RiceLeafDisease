import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.optim.lr_scheduler import LambdaLR


class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        # Convert logits to probabilities
        probs = F.softmax(inputs, dim=1)

        # Gather the probabilities corresponding to the target class
        targets_one_hot = F.one_hot(targets, num_classes=inputs.size(1))  # Shape: [batch_size, num_classes]
        pt = (probs * targets_one_hot).sum(dim=1)  # Shape: [batch_size]

        # Compute the focal loss
        focal_weight = self.alpha * (1 - pt) ** self.gamma
        loss = -focal_weight * torch.log(pt + 1e-8)

        # Apply reduction
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


def get_warmup_scheduler(optimizer, warm_up_steps, base_lr):
    # Set initial learning rate to base_lr / 10
    for param_group in optimizer.param_groups:
        param_group['lr'] = base_lr / 10

    def warmup_lr_scheduler(step):
        if step < warm_up_steps:
            return step / warm_up_steps  # Gradually scale up
        return 1.0  # Return the base_lr afterward

    return LambdaLR(optimizer, lr_lambda=warmup_lr_scheduler)