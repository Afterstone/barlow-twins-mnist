import logging
import pickle
import sys
import typing as t
from copy import deepcopy
from functools import partial
from pathlib import Path

import lightning.pytorch as pl
import torch as T
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger
from torch.utils.data import DataLoader, Dataset, TensorDataset
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor

import optuna

from .augmentations import MaskTensor, apply_random_gaussian_noise
from .models import LightningBarlowTwins


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
) -> tuple[T.Tensor, T.Tensor]:
    embs: T.Tensor = T.empty((0, emb_dim_size))
    targets: T.Tensor = T.empty((0, 1))
    for X, y in data_loader:
        X, y = X.to(device), y

        for aug in augmentations:
            X = aug(X)

        y_hat = encoder(X)
        embs = T.vstack((embs, y_hat.cpu()))
        targets = T.vstack((targets, y.cpu().unsqueeze(1)))
    return embs, targets.ravel()


def train_barlow_twins(
    trial: optuna.Trial,
    study_name: str,
    train_ds: Dataset,
    val_ds: Dataset,
    test_ds: Dataset,
) -> float:
    batch_size: int = trial.suggest_int("batch_size", 256, 4096, log=True)
    emb_dim_size: int = trial.suggest_int("emb_dim_size", 2, 64, log=True)
    l1_loss_weight = trial.suggest_float("l1_loss_weight", 0.1, 5000.0, log=True)
    l2_loss_weight = trial.suggest_float("l2_loss_weight", 0.1, 5000.0, log=True)
    aug_noise_sigma: float = trial.suggest_float("aug_noise_sigma", 0.0, 0.5)
    target_lr = trial.suggest_float("lr_target", 1e-5, 1e-3, log=True)
    online_lr = trial.suggest_float("lr_online", 1e-3, 1e-1, log=True)
    lambda_: float = trial.suggest_float("lambda", 0.0, 1.0)
    masktensor_prob: float = trial.suggest_float("masktensor_prob", 0.0, 1.0)
    masktensor_block_size: int = trial.suggest_int("masktensor_block_size", 0, 28)

    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_dl = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_dl = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    bt = LightningBarlowTwins(
        emb_dim_size=emb_dim_size,
        target_lr=target_lr,
        online_lr=online_lr,
        l1_loss_weight=l1_loss_weight,
        l2_loss_weight=l2_loss_weight,
        lambda_=lambda_,
        augmentations=[
            MaskTensor(mask_prob=masktensor_prob, block_size=masktensor_block_size),
            partial(apply_random_gaussian_noise, sigma=aug_noise_sigma, p=1.0),
        ],
    ).to("cuda")

    logger = TensorBoardLogger("lightning_logs", name=f"{study_name}")
    trainer = pl.Trainer(
        max_epochs=100,
        accelerator="gpu",
        devices="auto",
        logger=logger,
        check_val_every_n_epoch=1,
        callbacks=[
            ModelCheckpoint(
                monitor="val_logreg_loss",
                mode="min",
                dirpath="checkpoints",
                filename=f"{study_name}_{{epoch:02d}}-{{val_f1:.2f}}"
            ),
            EarlyStopping(monitor="val_logreg_loss", mode="min", patience=5),
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
        X_train, y_train = train_embs, train_targets.long()

        test_embs, test_targets = get_embeds(device, emb_dim_size, test_dl, encoder, augmentations)
        X_test, y_test = test_embs, test_targets.long()

        res = bt.evaluate(X_train, y_train, X_test, y_test)

    return res.loss


def main():
    train_ds, val_ds, train_mean, train_std = get_train_val()
    test_ds = get_test(mean=train_mean, std=train_std)

    study_name = "barlow_twins_seminar"
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

    objective = partial(
        train_barlow_twins,
        train_ds=train_ds,
        val_ds=val_ds,
        test_ds=test_ds,
        study_name=study_name,
    )

    optuna.logging.get_logger("optuna").addHandler(logging.StreamHandler(sys.stdout))
    study = optuna.create_study(
        direction=optuna.study.StudyDirection.MINIMIZE,
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
