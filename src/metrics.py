import torch 

def shifts_squared_loss(y_pred, y_true):
    true_shifts = y_true[:, 1:]
    pred_shifts = y_pred[:, 1:]
    # find nonzero cells for true tracks
    nz_idx = torch.nonzero(y_true[:, 0])
    # extract only shifts corresponding to true tracks
    true_shifts = true_shifts[nz_idx[:, 0], :, nz_idx[:, 1], nz_idx[:, 2]]
    pred_shifts = pred_shifts[nz_idx[:, 0], :, nz_idx[:, 1], nz_idx[:, 2]]
    # compute error
    squared_errors = torch.pow(true_shifts - pred_shifts, 2)
    # number of elements
    N = torch.mul(*squared_errors.size())
    loss = squared_errors.sum() / N
    return loss.item()


def precision(y_pred, y_true):
    true_probs = y_true[:, 0]
    pred_probs = y_pred[:, 0]
    # threshold predicted probabilities
    pred_probs = (pred_probs > 0.5).float()
    # find cells predicted as true tracks
    nz_idx = torch.nonzero(pred_probs)
    N = len(nz_idx)
    
    if N == 0:
        # zero precision
        return N
      
    # how much of them are really true tracks
    tp = true_probs[nz_idx[:, 0], nz_idx[:, 1], nz_idx[:, 2]].sum()
    return (tp / N).item()


def recall(y_pred, y_true):
    true_probs = y_true[:, 0]
    pred_probs = y_pred[:, 0]
    # threshold predicted probabilities
    pred_probs = (pred_probs > 0.5).float()
    # find true tracks' cells
    nz_idx = torch.nonzero(true_probs)
    # how much of them were predicted as true
    tp = pred_probs[nz_idx[:, 0], nz_idx[:, 1], nz_idx[:, 2]].sum()
    N = len(nz_idx)
    return (tp / N).item()


def nonzero_preds(y_pred, y_true):
    pred_probs = y_pred[:, 0]
    pred_probs = (pred_probs > 0.5).float().sum()
    return pred_probs.item()