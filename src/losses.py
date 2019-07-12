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


class LootLoss(torch.nn.Module):
    def __init__(self):
        super(LootLoss, self).__init__()
        self.probs_loss = torch.nn.BCELoss()
        self.shifts_loss = ShiftsMSELoss()

    def forward(self, inputs, target):
        probs_loss_value = self.probs_loss(inputs, target)
        shifts_loss_value = self.shifts_loss(inputs, target)
        loss = probs_loss_value + shifts_loss_value
        return loss