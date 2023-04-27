import torch as T
import lightning.pytorch as pl

from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor

from torch.utils.data import DataLoader, TensorDataset

# Import logistic regression
from .models import LightningBarlowTwins
import torchvision.transforms as trfs  # noqa: F401
from functools import partial
from .augmentations import apply_random_gaussian_noise
from lightning.pytorch.loggers import TensorBoardLogger


def get_train_val() -> tuple[TensorDataset, TensorDataset]:
    train_dataset = MNIST('data', train=True, download=True, transform=ToTensor())
    train_loader = DataLoader(train_dataset, batch_size=len(train_dataset), shuffle=True)

    X, y = next(iter(train_loader))
    perm = T.randperm(len(train_dataset))
    X, y = X[perm], y[perm]

    split_idx = int(len(train_dataset) * 0.8)

    X_train, y_train = X[:split_idx].cuda(), y[:split_idx].cuda()
    X_val, y_val = X[split_idx:].cuda(), y[split_idx:].cuda()
    mean, std = X_train.mean(), X_train.std()

    X_train = (X_train - mean) / std
    X_val = (X_val - mean) / std

    train_ds = TensorDataset(X_train, y_train)
    val_ds = TensorDataset(X_val, y_val)

    return train_ds, val_ds


def get_test() -> TensorDataset:
    test_dataset = MNIST('data', train=False, download=True, transform=ToTensor())
    test_loader = DataLoader(test_dataset, batch_size=len(test_dataset), shuffle=True)
    X, y = next(iter(test_loader))
    return TensorDataset(X.cuda(), y.cuda())


train_ds, val_ds = get_train_val()
test_ds = get_test()

train_dl = DataLoader(train_ds, batch_size=512, shuffle=True)
val_dl = DataLoader(val_ds, batch_size=512, shuffle=False)
test_dl = DataLoader(test_ds, batch_size=512, shuffle=False)

bt = LightningBarlowTwins(
    emb_dim_size=64,
    l1_loss_weight=100,
    l2_loss_weight=100,
    augmentations=[
        partial(apply_random_gaussian_noise, sigma=0.1, p=1.0),
        # trfs.RandomApply([trfs.GaussianBlur(kernel_size=3, sigma=(0.01, 1.0))], 0.5),
    ],
).to("cuda")

logger = TensorBoardLogger("lightning_logs", name="hallo")
trainer = pl.Trainer(
    max_epochs=100,
    accelerator="gpu",
    devices="auto",
    logger=logger
)
trainer.fit(model=bt, train_dataloaders=train_dl, val_dataloaders=val_dl)
