import torch
import torch.nn as nn
import torch.nn.functional as F

class CapsuleLoss(nn.Module):
    def __init__(self, lam_recon):
        super(CapsuleLoss, self).__init__()
        self.lam_recon = lam_recon

    def margin_loss(self, y_true, y_pred):
        y_true = y_true.unsqueeze(2).expand_as(y_pred)

        left = F.relu(0.9 - y_pred).pow(2)
        right = F.relu(y_pred - 0.1).pow(2)
        loss = y_true * left + 0.5 * (1.0 - y_true) * right
        return loss.sum(dim=1).mean()

    def reconstruction_loss(self, x_recon, x_true):
        return F.mse_loss(x_recon, x_true, reduction='mean')

    def forward(self, y_true, y_pred, x_recon, x_true):
        margin_loss = self.margin_loss(y_true, y_pred)
        recon_loss = self.reconstruction_loss(x_recon, x_true)
        return margin_loss + self.lam_recon * recon_loss
