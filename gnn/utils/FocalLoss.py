# Source: https://medium.com/data-scientists-diary/implementing-focal-loss-in-pytorch-for-class-imbalance-24d8aa3b59d9
import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalCrossEntropyLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2, reduction='mean'):
        super(FocalCrossEntropyLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        # Standard CrossEntropyLoss
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        
        # Calculate probabilities
        probs = torch.softmax(inputs, dim=1)
        
        # Get the probabilities of the correct class
        p_t = probs.gather(1, targets.view(-1, 1)).squeeze()

        # Compute the Focal Loss adjustment
        focal_loss = self.alpha * (1 - p_t) ** self.gamma * ce_loss

        # Reduce loss based on the specified reduction method
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

