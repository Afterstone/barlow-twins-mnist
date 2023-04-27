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

from .augmentations import apply_random_gaussian_noise
# Import logistic regression
from .models import LightningBarlowTwins
import pickle
from pathlib import Path


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
    target_lr = trial.suggest_float("target_lr", 1e-6, 1e-0, log=True)
    online_lr = trial.suggest_float("projector_lr", 1e-6, 1e-0, log=True)

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
        max_epochs=20,
        accelerator="gpu",
        devices="auto",
        logger=logger,
        # callbacks=[PyTorchLightningPruningCallback(trial, monitor="val_f1")],
    )
    trainer.fit(model=bt, train_dataloaders=train_dl, val_dataloaders=val_dl)

    device = bt.device
    encoder = deepcopy(bt.target.encoder).to(device)
    augmentations = bt.augmentations
    with T.no_grad():
        encoder.eval()

        train_emb: np.ndarray = np.empty((0, emb_dim_size))
        train_targets: np.ndarray = np.empty((0, 1))
        for X_train, y_train in train_dl:
            X_train, y_train = X_train.to(device), y_train.to(device)

            for aug in augmentations:
                X_train = aug(X_train)

            y_hat = encoder(X_train)
            train_emb = np.vstack((train_emb, y_hat.cpu().detach().numpy()))
            train_targets = np.vstack((train_targets, y_train.cpu().detach().numpy().reshape(-1, 1)))

        train_emb_mean, train_emb_std = train_emb.mean(axis=0), train_emb.std(axis=0)
        train_emb = (train_emb - train_emb_mean) / train_emb_std

        lr = LogisticRegression(max_iter=1000)
        lr.fit(train_emb, train_targets.ravel())

        emb_test: np.ndarray = np.empty((0, emb_dim_size))
        emb_targets: np.ndarray = np.empty((0, 1))
        for X_test, y_test in test_dl:
            X_test, y_test = X_test.to(device), y_test.to(device)

            for aug in augmentations:
                X_test = aug(X_test)

            y_hat = encoder(X_test)
            emb_test = np.vstack((emb_test, y_hat.cpu().detach().numpy()))
            emb_targets = np.vstack((emb_targets, y_test.cpu().detach().numpy().reshape(-1, 1)))

        emb_test = (emb_test - train_emb_mean) / train_emb_std

    y_hat = lr.predict(emb_test)
    # test_acc = accuracy_score(emb_targets, y_hat)
    test_f1 = f1_score(emb_targets, y_hat, average='weighted')

    return float(test_f1)


def main():
    train_ds, val_ds = get_train_val()
    test_ds = get_test()

    train_dl = DataLoader(train_ds, batch_size=512, shuffle=True)
    val_dl = DataLoader(val_ds, batch_size=512, shuffle=False)
    test_dl = DataLoader(test_ds, batch_size=512, shuffle=False)
    objective = partial(train_barlow_twins, train_dl=train_dl, val_dl=val_dl, test_dl=test_dl)

    optuna.logging.get_logger("optuna").addHandler(logging.StreamHandler(sys.stdout))
    study_name = "barlow_twins"
    storage_name = "sqlite:///{}.db".format(study_name)
    sampler_path = Path(f'{study_name}.pkl')
    if not sampler_path.exists():
        sampler = optuna.samplers.TPESampler(multivariate=True)
        with open(sampler_path, 'wb') as f:
            pickle.dump(sampler, f)
    else:
        with open(sampler_path, 'rb') as f:
            sampler = pickle.load(f)

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
