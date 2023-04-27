import logging
import sys
from copy import deepcopy
from functools import partial

import lightning.pytorch as pl
import numpy as np
import optuna
import torch as T
import torchvision.transforms as trfs  # noqa: F401
from lightning.pytorch.loggers import TensorBoardLogger
from sklearn.linear_model import LogisticRegression
# Import accuracy score
from sklearn.metrics import accuracy_score, f1_score  # noqa: F401
from torch.utils.data import DataLoader, TensorDataset
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
from lightning.pytorch.callbacks import ModelCheckpoint

from .augmentations import apply_random_gaussian_noise
# Import logistic regression
from .models import LightningBarlowTwins
import pickle
from pathlib import Path
import typing as t


def get_train_val(train_val_split: float = 0.8) -> tuple[TensorDataset, TensorDataset, float, float]:
    train_dataset = MNIST('data', train=True, download=True, transform=ToTensor())
    train_loader = DataLoader(train_dataset, batch_size=len(train_dataset), shuffle=False)

    X, y = next(iter(train_loader))
    perm = T.randperm(len(train_dataset))
    X, y = X[perm], y[perm]

    split_idx = int(len(train_dataset) * train_val_split)

    X_train, y_train = X[:split_idx].cuda(), y[:split_idx].cuda()
    X_val, y_val = X[split_idx:].cuda(), y[split_idx:].cuda()
    mean, std = X_train.mean(), X_train.std()

    X_train = (X_train - mean) / std
    X_val = (X_val - mean) / std

    train_ds = TensorDataset(X_train, y_train)
    val_ds = TensorDataset(X_val, y_val)

    return train_ds, val_ds, mean, std


def get_test(
    mean: float,
    std: float,
) -> TensorDataset:
    test_dataset = MNIST('data', train=False, download=True, transform=ToTensor())
    test_loader = DataLoader(test_dataset, batch_size=len(test_dataset), shuffle=False)
    X, y = next(iter(test_loader))
    X, y = X.cuda(), y.cuda()
    X = (X - mean) / std
    return TensorDataset(X.cuda(), y.cuda())


def get_embeds(
    device: str | T.device,
    emb_dim_size: int,
    data_loader: DataLoader,
    encoder: T.nn.Module,
    augmentations: list[T.nn.Module | t.Callable[..., T.Tensor]],
) -> tuple[np.ndarray, np.ndarray]:
    embs: np.ndarray = np.empty((0, emb_dim_size))
    targets: np.ndarray = np.empty((0, 1))
    for X, Y in data_loader:
        X, Y = X.to(device), Y.to(device)

        for aug in augmentations:
            X = aug(X)

        y_hat = encoder(X)
        embs = np.vstack((embs, y_hat.cpu().detach().numpy()))
        targets = np.vstack((targets, Y.cpu().detach().numpy().reshape(-1, 1)))
    return embs, targets.ravel()


def train_barlow_twins(
    trial: optuna.Trial,
    train_dl: DataLoader,
    val_dl: DataLoader,
    test_dl: DataLoader,
) -> float:
    emb_dim_size = 32
    l1_loss_weight = trial.suggest_float("l1_loss_weight", 0.1, 10000.0, log=True)
    l2_loss_weight = trial.suggest_float("l2_loss_weight", 0.1, 10000.0, log=True)
    aug_noise_sigma = trial.suggest_float("aug_noise_sigma", 1e-4, 1e-0, log=True)
    target_lr = trial.suggest_float("lr_target", 1e-6, 1e-0, log=True)
    online_lr = trial.suggest_float("lr_online", 1e-6, 1e-0, log=True)

    bt = LightningBarlowTwins(
        emb_dim_size=emb_dim_size,
        target_lr=target_lr,
        online_lr=online_lr,
        l1_loss_weight=l1_loss_weight,
        l2_loss_weight=l2_loss_weight,
        augmentations=[
            partial(apply_random_gaussian_noise, sigma=aug_noise_sigma, p=1.0),
            # trfs.RandomApply([trfs.GaussianBlur(kernel_size=3, sigma=(0.01, 1.0))], 0.5),
        ],
    ).to("cuda")

    logger = TensorBoardLogger("lightning_logs", name="hallo")
    trainer = pl.Trainer(
        max_epochs=10,
        accelerator="gpu",
        devices="auto",
        logger=logger,
        callbacks=[
            ModelCheckpoint(monitor="val_f1", mode="max", dirpath="checkpoints"),
            # PyTorchLightningPruningCallback(trial, monitor="val_f1")
        ],
    )
    trainer.fit(
        model=bt,
        train_dataloaders=train_dl,
        val_dataloaders=val_dl,
    )

    device = bt.device
    encoder = deepcopy(bt.target.encoder).to(device)
    augmentations = bt.augmentations
    with T.no_grad():
        encoder.eval()

        train_embs, train_targets = get_embeds(device, emb_dim_size, train_dl, encoder, augmentations)
        test_embs, test_targets = get_embeds(device, emb_dim_size, test_dl, encoder, augmentations)

        lr = LogisticRegression(max_iter=1000)
        lr.fit(train_embs, train_targets.ravel())

        test_f1 = f1_score(test_targets, lr.predict(test_embs).ravel(), average='weighted')

    return float(test_f1)


def main():
    train_ds, val_ds, train_mean, train_std = get_train_val()
    test_ds = get_test(mean=train_mean, std=train_std)

    train_dl = DataLoader(train_ds, batch_size=512, shuffle=True)
    val_dl = DataLoader(val_ds, batch_size=512, shuffle=False)
    test_dl = DataLoader(test_ds, batch_size=512, shuffle=False)
    objective = partial(train_barlow_twins, train_dl=train_dl, val_dl=val_dl, test_dl=test_dl)

    study_name = "barlow_twins"
    optuna_dir = Path("optuna")
    optuna_dir.mkdir(parents=True, exist_ok=True)
    storage_name = f"sqlite:///{optuna_dir}/{study_name}.db"
    sampler_path = Path(f'./{optuna_dir}/{study_name}.pkl')
    if not sampler_path.exists():
        sampler = optuna.samplers.TPESampler(multivariate=True)
        with open(sampler_path, 'wb') as f:
            pickle.dump(sampler, f)
    else:
        with open(sampler_path, 'rb') as f:
            sampler = pickle.load(f)

    optuna.logging.get_logger("optuna").addHandler(logging.StreamHandler(sys.stdout))
    study = optuna.create_study(
        direction="maximize",
        storage=storage_name,
        study_name=study_name,
        load_if_exists=True,
        sampler=sampler,
    )
    study.optimize(
        objective,
        timeout=4 * 60 * 60,
        show_progress_bar=True,
        catch=(ValueError,),
    )


if __name__ == "__main__":
    main()
