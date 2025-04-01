import os

from lightning.pytorch.callbacks import Callback, TQDMProgressBar


class StepProgressBar(TQDMProgressBar):
    def __init__(self):
        super().__init__()
        self._global_step = 0

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        super().on_train_batch_start(trainer, pl_module, batch, batch_idx)
        self._global_step = trainer.global_step

    def get_metrics(self, trainer, pl_module):
        items = super().get_metrics(trainer, pl_module)
        items["step"] = self._global_step
        return items


class RecentCheckpointsCallback(Callback):
    def __init__(self, save_path, save_top_k=5, save_every_steps=5000):
        self.save_path = save_path
        self.save_top_k = save_top_k
        self.filename = "checkpoint-step={step}"
        self.saved_checkpoints = []
        self.save_every_steps = save_every_steps

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if trainer.global_step % self.save_every_steps == 0:
            checkpoint_path = os.path.join(
                self.save_path,
                self.filename.format(step=trainer.global_step) + ".ckpt"
            )
            trainer.save_checkpoint(checkpoint_path)
            self.saved_checkpoints.append(checkpoint_path)

            if len(self.saved_checkpoints) > self.save_top_k:
                oldest_checkpoint = self.saved_checkpoints.pop(0)
                if os.path.exists(oldest_checkpoint):
                    os.remove(oldest_checkpoint)
