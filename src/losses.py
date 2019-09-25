import torch
import torch.nn as nn
import torch.nn.functional as F


class ShiftsMSELoss(torch.nn.Module):
    def __init__(self):
        super(ShiftsMSELoss, self).__init__()
    
    def forward(self, inputs, target):
        true_shifts = target[:, 1:]
        pred_shifts = inputs[:, 1:]
        # find nonzero cells for true tracks
        nz_idx = torch.nonzero(target[:, 0])
        # extract only shifts corresponding to true tracks
        true_shifts = true_shifts[nz_idx[:, 0], :, nz_idx[:, 1], nz_idx[:, 2]]
        pred_shifts = pred_shifts[nz_idx[:, 0], :, nz_idx[:, 1], nz_idx[:, 2]]
        # compute error
        squared_errors = torch.pow(true_shifts - pred_shifts, 2)
        # number of elements
        N = torch.mul(*squared_errors.size())
        loss = squared_errors.sum() / N
        return loss


class FakesLoss(torch.nn.Module):
    def __init__(self):
        super(FakesLoss, self).__init__()
        self.bce_loss = torch.nn.BCELoss()

    def forward(self, inputs, target):
        true_mask = target[:, 0]
        pred_mask = inputs[:, 0]
        # find outputs predicted as true tracks
        nz_idx = torch.nonzero(pred_mask)
        # extract corresponding cells' values
        y_true = true_mask[nz_idx[:, 0], nz_idx[:, 1], nz_idx[:, 2]]
        y_pred = pred_mask[nz_idx[:, 0], nz_idx[:, 1], nz_idx[:, 2]]
        return self.bce_loss(y_pred, y_true)


class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, logits=False, reduce=True):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.logits = logits
        self.reduce = reduce

    def forward(self, inputs, targets):
        if self.logits:
            BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduce=False)
        else:
            BCE_loss = F.binary_cross_entropy(inputs, targets, reduce=False)
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1-pt)**self.gamma * BCE_loss

        if self.reduce:
            return torch.mean(F_loss)
        else:
            return F_loss


class LootLoss(torch.nn.Module):
    def __init__(self):
        super(LootLoss, self).__init__()
        self.probs_loss = FocalLoss(gamma=2, alpha=0.9)
        self.shifts_loss = ShiftsMSELoss()

    def forward(self, inputs, target):
        probs_loss_value = self.probs_loss(inputs[:, 0], target[:, 0])
        shifts_loss_value = self.shifts_loss(inputs, target)
        loss = probs_loss_value + shifts_loss_value
        return loss