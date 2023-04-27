
import torch as T


class CrossCorrelationLoss(T.nn.Module):
    def __init__(self, lambda_: float = 1.0, epsilon: float = 1e-16):
        super().__init__()

        self.lambda_ = lambda_
        self.epsilon = epsilon

    def forward(self, z1: T.Tensor, z2: T.Tensor):
        if not z1.shape == z2.shape:
            raise ValueError(f"z1 and z2 must have the same shape, got {z1.shape} and {z2.shape}.")

        device = z1.device
        z1_norm = (z1 - z1.mean(0)) / (z1.std(0) + self.epsilon)
        z2_norm = (z2 - z2.mean(0)) / (z2.std(0) + self.epsilon)

        c = (z1_norm.T @ z2_norm) / z1.shape[0]
        eye = T.eye(c.shape[0], device=device, dtype=T.float32)
        c_diff = (c - eye) ** 2
        mask = T.ones_like(c_diff, device=device) - T.eye(c_diff.size(0), dtype=T.float32, device=device)
        c_diff = c_diff + c_diff * mask * (self.lambda_ - 1)
        loss = c_diff.sum()
        return loss
