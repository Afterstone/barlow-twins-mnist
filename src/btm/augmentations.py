import torch as T


def apply_random_gaussian_noise(x: T.Tensor, sigma: float = 0.1, p: float = 1.0) -> T.Tensor:
    if T.rand(1).item() < p:
        return x + T.randn_like(x) * sigma
    else:
        return x
