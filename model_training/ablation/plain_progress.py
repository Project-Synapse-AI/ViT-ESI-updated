"""Opt-in, read-only training progress for piped notebook runs."""

import json
import time
from pathlib import Path

from pytorch_lightning.callbacks import Callback


class PlainProgress(Callback):
    def __init__(self, path):
        self.path = Path(path)
        self.started = time.monotonic()

    def emit(self, trainer, event, **extra):
        if not trainer.is_global_zero:
            return
        record = dict(event=event, epoch=min(int(trainer.current_epoch) + 1, trainer.max_epochs),
                      max_epochs=trainer.max_epochs, step=int(trainer.global_step),
                      elapsed_s=round(time.monotonic() - self.started, 1), **extra)
        line = json.dumps(record)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as stream:
            stream.write(line + "\n")
        print("TRAIN_PROGRESS " + line, flush=True)

    def on_train_start(self, trainer, pl_module):
        self.emit(trainer, "train_start", batches=trainer.num_training_batches)

    def on_train_epoch_start(self, trainer, pl_module):
        self.epoch_started = time.monotonic()
        self.emit(trainer, "epoch_start")

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if batch_idx == 0 or (batch_idx + 1) % 100 == 0:
            elapsed = time.monotonic() - self.epoch_started
            self.emit(trainer, "batch", batch=batch_idx + 1,
                      batches=trainer.num_training_batches,
                      batches_per_s=round((batch_idx + 1) / max(elapsed, 1e-9), 3))

    def on_validation_end(self, trainer, pl_module):
        if trainer.sanity_checking:
            return
        metrics = {name: float(value.detach().cpu())
                   for name, value in trainer.callback_metrics.items()
                   if name == "validation_loss"}
        elapsed = time.monotonic() - self.epoch_started
        self.emit(trainer, "validation_end", epoch_s=round(elapsed, 1),
                  remaining_training_s_estimate=round(
                      elapsed * max(0, trainer.max_epochs - trainer.current_epoch - 1)),
                  metrics=metrics)

    def on_train_end(self, trainer, pl_module):
        self.emit(trainer, "train_end")
