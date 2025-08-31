import torch
def low_high_loss(prediction, target, base_loss_fn):
    # --- low high constraints ---
    low = prediction[:, 2, :]
    high = prediction[:, 3, :]

    # low should be non-positive, so we are punishing the positive parts
    positive_low = torch.relu(low)

    # high should be non-negative, so we are punishing the negative parts
    negative_high = torch.relu(-1 * high)

    low_high_loss = (negative_high + positive_low).mean()
    return low_high_loss