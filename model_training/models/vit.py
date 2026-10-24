"""
Transformer (ViT-style) model for EEG -> source time series.

Input:  EEG batch shaped (B, E, T)
Output: Source batch shaped (B, S, T)

Implementation: treat each time step as a token whose features are the sensors.
This is "ViT-like" in the sense of using a Transformer encoder with learnable positional
embeddings, but is tailored to EEG time series.
"""

from __future__ import annotations

import torch
from torch import nn
import pytorch_lightning as pl


class FixedAveragingLayer(nn.Module):
    """Preserve a pre-norm encoder layer except for its attention branch."""

    def __init__(self, layer: nn.TransformerEncoderLayer, mixer: str, radius: int):
        super().__init__()
        self.mixer = mixer
        self.radius = radius
        # Reuse initialized modules; replacing attention must not consume RNG.
        for name in ("norm1", "norm2", "linear1", "linear2", "dropout", "dropout1", "dropout2"):
            setattr(self, name, getattr(layer, name))
        self.activation = layer.activation

    def mix(self, x: torch.Tensor) -> torch.Tensor:
        if self.mixer == "mean":
            return x.mean(dim=1, keepdim=True).expand_as(x)
        # Pool independently per feature; exclude zero padding from the divisor.
        return torch.nn.functional.avg_pool1d(
            x.transpose(1, 2), kernel_size=2 * self.radius + 1,
            stride=1, padding=self.radius, count_include_pad=False,
        ).transpose(1, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.dropout1(self.mix(self.norm1(x)))
        return x + self.dropout2(self.linear2(self.dropout(self.activation(self.linear1(self.norm2(x))))))


class FixedAveragingEncoder(nn.Module):
    def __init__(self, encoder: nn.TransformerEncoder, mixer: str, radius: int):
        super().__init__()
        self.layers = nn.ModuleList([
            FixedAveragingLayer(layer, mixer, radius) for layer in encoder.layers
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return x


class EEGViT(nn.Module):
    def __init__(
        self,
        num_sensor: int,
        num_source: int,
        n_times: int = 500,
        embed_dim: int = 256,
        depth: int = 6,
        num_heads: int = 8,
        mlp_dim: int = 512,
        dropout: float = 0.1,
        mixer: str = "attention",
        window_radius: int = 5,
    ) -> None:
        super().__init__()
        if mixer not in ("attention", "mean", "window"):
            raise ValueError(f"Unknown timepoint mixer: {mixer}")
        if window_radius < 0 or (mixer == "window" and window_radius >= n_times):
            raise ValueError("window_radius must be nonnegative and smaller than n_times for window mixing")
        self.num_sensor = num_sensor
        self.num_source = num_source
        self.n_times = n_times

        self.in_proj = nn.Linear(num_sensor, embed_dim)
        self.pos_embed = nn.Parameter(torch.zeros(1, n_times, embed_dim))
        self.pos_drop = nn.Dropout(dropout)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=mlp_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=depth)
        self.out_proj = nn.Linear(embed_dim, num_source)

        self._reset_parameters()
        if mixer != "attention":
            self.encoder = FixedAveragingEncoder(self.encoder, mixer, window_radius)

    def _reset_parameters(self) -> None:
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        # in/out projections use PyTorch defaults (Kaiming/uniform) which are fine.

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, E, T)
        returns: (B, S, T)
        """
        # (B, T, E)
        x = x.permute(0, 2, 1)
        if x.shape[1] != self.n_times:
            raise ValueError(
                f"EEGViT was initialized with n_times={self.n_times}, got T={x.shape[1]}. "
                "Use the same n_times for training/eval or reinitialize the model."
            )
        x = self.in_proj(x)  # (B, T, D)
        x = self.pos_drop(x + self.pos_embed)
        x = self.encoder(x)  # (B, T, D)
        x = self.out_proj(x)  # (B, T, S)
        return x.permute(0, 2, 1)  # (B, S, T)


class EEGViTpl(pl.LightningModule):
    def __init__(
        self,
        num_sensor: int,
        num_source: int,
        n_times: int = 500,
        embed_dim: int = 256,
        depth: int = 6,
        num_heads: int = 8,
        mlp_dim: int = 512,
        dropout: float = 0.1,
        optimizer=torch.optim.Adam,
        lr: float = 1e-3,
        criterion=torch.nn.MSELoss(),
        mixer: str = "attention",
        window_radius: int = 5,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(ignore=["criterion", "optimizer"])

        self.optimizer = optimizer
        self.lr = lr
        self.criterion = criterion

        self.model = EEGViT(
            num_sensor=num_sensor,
            num_source=num_source,
            n_times=n_times,
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
            mlp_dim=mlp_dim,
            dropout=dropout,
            mixer=mixer,
            window_radius=window_radius,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def configure_optimizers(self):
        return self.optimizer(self.model.parameters(), lr=self.lr)

    def training_step(self, batch, batch_idx):
        eeg, src = batch
        eeg = eeg.float()
        src = src.float()
        src_hat = self.forward(eeg)
        loss = self.criterion(src_hat, src)
        self.log("train_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        eeg, src = batch
        eeg = eeg.float()
        src = src.float()
        src_hat = self.forward(eeg)
        loss = self.criterion(src_hat, src)
        self.log("validation_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        return loss

