import torch as T
import torch.nn.functional as F


def apply_random_gaussian_noise(x: T.Tensor, sigma: float = 0.1, p: float = 1.0) -> T.Tensor:
    if T.rand(1).item() < p:
        return x + T.randn_like(x) * sigma
    else:
        return x


class MaskTensor(T.nn.Module):
    def __init__(self, block_size: int, mask_prob: float):
        super().__init__()

        self.block_size = block_size
        self.mask_prob = mask_prob

    def forward(self, x: T.Tensor) -> T.Tensor:
        if x.ndim == 3:
            x = x.unsqueeze(1)
        # Calculate the size of the downsampled mask
        w_downsized = x.shape[2] // self.block_size
        h_downsized = x.shape[3] // self.block_size

        # Create a random mask with the given probability
        small_mask = T.where(T.rand((w_downsized, h_downsized)) < self.mask_prob, 0, 1).float()

        # Upsample the mask to match the input size
        upsampled_mask = F.interpolate(small_mask.unsqueeze(0).unsqueeze(0), size=(x.shape[2], x.shape[3]), mode='nearest')

        # Repeat the mask across    the channel dimension
        upsampled_mask = upsampled_mask.repeat(x.shape[0], x.shape[1], 1, 1)

        # Apply the mask to the input tensor
        upsampled_mask = upsampled_mask.to(x.device)
        masked_x = x * upsampled_mask

        return masked_x
