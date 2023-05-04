import typing as t

import lightning.pytorch as pl
import torch as T
from sklearn.metrics import accuracy_score, f1_score

from .losses import CrossCorrelationLoss
from sklearn.linear_model import LogisticRegression


class Autoencoder(T.nn.Module):
    def __init__(self, emb_dim_size: int = 10):
        super().__init__()

        self.emb_dim_size = emb_dim_size

        self.encoder = T.nn.Sequential(
            T.nn.BatchNorm2d(1),
            T.nn.Conv2d(1, 8, 3, stride=1, padding=1),
            T.nn.LeakyReLU(),
            T.nn.MaxPool2d(2),
            T.nn.BatchNorm2d(8),
            T.nn.Conv2d(8, 16, 3, stride=1, padding=1),
            T.nn.LeakyReLU(),
            T.nn.MaxPool2d(2),
            T.nn.BatchNorm2d(16),
            T.nn.Flatten(),
            T.nn.Linear(7 * 7 * 16, self.emb_dim_size),
            T.nn.BatchNorm1d(self.emb_dim_size),
            T.nn.Sigmoid(),
        )
        self.decoder = T.nn.Sequential(
            T.nn.Linear(self.emb_dim_size, 7 * 7 * 16),
            T.nn.LeakyReLU(),
            T.nn.BatchNorm1d(7 * 7 * 16),
            T.nn.Linear(7 * 7 * 16, 7 * 7 * 32),
            T.nn.LeakyReLU(),
            T.nn.BatchNorm1d(7 * 7 * 32),
            T.nn.Sigmoid(),
        )

    def forward(self, x: T.Tensor) -> tuple[T.Tensor, T.Tensor]:
        z = self.encoder(x)
        x_hat = self.decoder(z)
        return x_hat, z


class LightningBarlowTwins(pl.LightningModule):
    def __init__(
        self,
        augmentations: list[T.nn.Module | t.Callable[..., T.Tensor]],
        emb_dim_size: int = 10,
        l1_loss_weight: float = 0.0,
        l2_loss_weight: float = 0.0,
        target_lr: float = 1e-3,
        online_lr: float = 1e-2,
        lambda_: float = 1.0,
    ):
        super().__init__()

        self.target_lr = target_lr
        self.online_lr = online_lr

        self.emb_dim_size = emb_dim_size
        self.augmentations = augmentations
        self.l1_loss_weight = l1_loss_weight
        self.l2_loss_weight = l2_loss_weight

        self.automatic_optimization = False
        self.online = Autoencoder(emb_dim_size)
        self.target = Autoencoder(emb_dim_size)

        self.loss_fn = CrossCorrelationLoss(lambda_=lambda_)

        self.train_embeddings: T.Tensor = T.empty((0, self.emb_dim_size))
        self.train_targets: T.Tensor = T.empty((0,))
        self.val_embeddings: T.Tensor = T.empty((0, self.emb_dim_size))
        self.val_targets: T.Tensor = T.empty((0,))

    def configure_optimizers(self):
        opt_target = T.optim.Adam(self.target.parameters(), lr=self.target_lr, weight_decay=1e-5)
        opt_online = T.optim.Adam(self.online.parameters(), lr=self.online_lr, weight_decay=1e-5)

        return [
            opt_online,
            opt_target,
        ]

    def on_train_epoch_start(self) -> None:
        self.train_embeddings = T.empty((0, self.emb_dim_size))
        self.train_targets = T.empty((0,))

    def training_step(self, batch: T.Tensor, batch_idx) -> T.Tensor:
        opt_online, opt_target = self.optimizers()  # type: ignore
        x, y = batch

        x1, x2 = x, x
        for aug in self.augmentations:
            x1 = aug(x1)
            x2 = aug(x2)

        x_hat_target, z1 = self.target(x1)
        x_hat_online, z2 = self.online(x2)

        end_loss = self.loss_fn(x_hat_target, x_hat_online)
        l1_loss = self.l1_loss_weight * T.mean(T.abs(z1) + T.abs(z2))
        l2_loss = self.l2_loss_weight * T.sqrt(T.mean(T.square(z1) + T.square(z2)))
        loss = end_loss + l1_loss + l2_loss

        self.manual_backward(loss)
        opt_online.step()
        opt_target.step()

        self.train_embeddings = T.vstack((self.train_embeddings, z1.detach().cpu()))
        self.train_targets = T.hstack((self.train_targets, y.detach().cpu()))

        self.log('train_loss', loss)
        self.log('train_l1_loss', l1_loss)
        self.log('train_l2_loss', l2_loss)

        return loss

    def on_train_epoch_end(self) -> None:
        if len(self.train_embeddings) > 0:
            X_train = self.train_embeddings.cpu().detach().numpy()
            y_train = self.train_targets.cpu().long().detach().numpy()

            lr = LogisticRegression(max_iter=1000, solver="newton-cholesky")
            lr.fit(X_train, y_train)
            y_hat = lr.predict(X_train)
            train_acc = accuracy_score(y_train, y_hat)
            train_f1 = f1_score(y_train, y_hat, average='weighted')

            self.log('train_acc', float(train_acc))
            self.log('train_f1', float(train_f1))

    def on_validation_epoch_start(self) -> None:
        self.val_embeddings = T.empty((0, self.emb_dim_size))
        self.val_targets = T.empty((0,))

    def validation_step(self, batch: T.Tensor, batch_idx) -> None:
        x, y = batch

        x1, x2 = x, x
        for aug in self.augmentations:
            x1 = aug(x1)
            x2 = aug(x2)

        x_hat_target, z1 = self.target(x1)
        x_hat_online, z2 = self.online(x2)

        end_loss = self.loss_fn(x_hat_target, x_hat_online)
        l1_loss = self.l1_loss_weight * T.mean(T.abs(z1) + T.abs(z2))
        l2_loss = self.l2_loss_weight * T.sqrt(T.mean(T.square(z1) + T.square(z2)))
        loss = end_loss + l1_loss + l2_loss

        self.val_embeddings = T.vstack((self.val_embeddings, z1.detach().cpu()))
        self.val_targets = T.hstack((self.val_targets, y.detach().cpu()))

        self.log('val_loss', loss)
        self.log('val_l1_loss', l1_loss)
        self.log('val_l2_loss', l2_loss)

    def on_validation_epoch_end(self) -> None:
        if len(self.val_embeddings) > 0 and len(self.train_embeddings) > 0:
            val_acc, val_f1, val_logreg_loss = self.evaluate_on_task(
                self.train_embeddings,
                self.train_targets,
                self.val_embeddings,
                self.val_targets,
            )
            self.log('val_acc', float(val_acc))
            self.log('val_f1', float(val_f1))
            self.log('val_logreg_loss', float(val_logreg_loss))

    def evaluate_on_task(
        self,
        X_train: T.Tensor,
        y_train: T.Tensor,
        X_val: T.Tensor,
        y_val: T.Tensor,
    ) -> tuple[float, float, float]:
        X_train_np = X_train.cpu().detach().numpy()
        y_train_np = y_train.cpu().long().detach().numpy()
        X_val_np = X_val.cpu().detach().numpy()
        y_val_np = y_val.cpu().long().detach().numpy()

        lr = LogisticRegression(max_iter=1000, solver="newton-cholesky")
        lr.fit(X_train_np, y_train_np)
        y_hat = lr.predict(X_val_np)

        val_acc = accuracy_score(y_val_np, y_hat)
        val_f1 = f1_score(y_val_np, y_hat, average='weighted')

        y_probs = lr.predict_proba(X_val_np)
        y_val_onehot = T.nn.functional.one_hot(y_val.long(), num_classes=10).float()
        logreg_loss = T.nn.functional.cross_entropy(T.tensor(y_probs), y_val_onehot)

        return float(val_acc), float(val_f1), float(logreg_loss)
