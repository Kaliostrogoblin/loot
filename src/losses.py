import torch

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



import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class FocalLoss(nn.Module):
    def __init__(self, gamma=0, alpha=None, size_average=True):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        if isinstance(alpha,(float,int)): self.alpha = torch.Tensor([alpha,1-alpha])
        if isinstance(alpha,list): self.alpha = torch.Tensor(alpha)
        self.size_average = size_average

    def forward(self, input, target):
        if input.dim()>2:
            input = input.view(input.size(0),input.size(1),-1)  # N,C,H,W => N,C,H*W
            input = input.transpose(1,2)    # N,C,H*W => N,H*W,C
            input = input.contiguous().view(-1,input.size(2))   # N,H*W,C => N*H*W,C
        target = target.view(-1,1)

        logpt = F.log_softmax(input)
        logpt = logpt.gather(1,target)
        logpt = logpt.view(-1)
        pt = Variable(logpt.data.exp())

        if self.alpha is not None:
            if self.alpha.type()!=input.data.type():
                self.alpha = self.alpha.type_as(input.data)
            at = self.alpha.gather(0,target.data.view(-1))
            logpt = logpt * Variable(at)

        loss = -1 * (1-pt)**self.gamma * logpt
        if self.size_average: return loss.mean()
        else: return loss.sum()



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