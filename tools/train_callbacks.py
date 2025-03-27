import os
import pathlib
from typing import Dict

import h5py
import numpy as np
import torch
import tqdm
from lightning.pytorch.callbacks import Callback, TQDMProgressBar

from evaluate import remove_ignored_phonemes
from tools import label
from tools.export_tool import Exporter
from tools.metrics import Metric, VlabelerEditRatio, BoundaryEditRatio, BoundaryEditRatioWeighted
from tools.post_processing import post_processing


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


class VlabelerEvaluateCallback(Callback):
    def __init__(self, binary_data_folder, out_tg_dir):
        super().__init__()
        self.binary_data_folder = pathlib.Path(binary_data_folder)
        self.out_tg_dir = pathlib.Path(out_tg_dir)
        self.dataset = []
        self.load_h5py_file()

    def load_h5py_file(self):
        h5py_file_path = str(pathlib.Path(self.binary_data_folder) / "evaluate.h5py")
        with h5py.File(h5py_file_path, "r") as h5py_file:
            items_group = h5py_file["items"]
            for item in items_group.values():
                input_feature = np.array(item["input_feature"])
                melspec = np.array(item["melspec"])
                wav_length = np.array(item["wav_length"])
                ph_seq = [ph.decode('utf-8') for ph in item["ph_seq"]]
                word_seq = [word.decode('utf-8') for word in item["word_seq"]]
                ph_idx_to_word_idx = np.array(item["ph_idx_to_word_idx"])
                wav_path = item["wav_path"][()].decode('utf-8')
                tg_path = item["tg_path"][()].decode('utf-8')
                self.dataset.append(
                    (input_feature, melspec, wav_length, ph_seq, word_seq, ph_idx_to_word_idx, wav_path, tg_path))

    def on_validation_start(self, trainer, pl_module):
        tg_paths = []
        predictions = []
        for batch in tqdm.tqdm(self.dataset, desc="evaluate_forward:", total=len(self.dataset)):
            input_feature, mel_spec, wav_length, ph_seq, word_seq, ph_idx_to_word_idx, wav_path, tg_path = batch
            tg_paths.append(tg_path)

            ph_seq, ph_intervals, word_seq, word_intervals, confidence, _, _ = trainer.model.infer_once(
                torch.from_numpy(input_feature).to('cuda'), mel_spec, wav_length, ph_seq, word_seq, ph_idx_to_word_idx,
                False, False)

            predictions.append((wav_path, wav_length, confidence, ph_seq, ph_intervals, word_seq, word_intervals,))

        predictions, log = post_processing(predictions)
        out_tg_dir = self.out_tg_dir / "evaluate" / str(trainer.global_step)
        exporter = Exporter(predictions, log, out_tg_dir)
        exporter.export(['textgrid'])

        iterable = list(out_tg_dir.rglob("*.TextGrid"))

        metrics: Dict[str, Metric] = {
            "BoundaryEditRatio": BoundaryEditRatio(),
            "BoundaryEditRatioWeighted": BoundaryEditRatioWeighted(),
            "VlabelerEditRatio10-20ms": VlabelerEditRatio(move_min=0.01, move_max=0.02),
            "VlabelerEditRatio20-50ms": VlabelerEditRatio(move_min=0.02, move_max=0.05),
            "VlabelerEditRatio50-100ms": VlabelerEditRatio(move_min=0.05, move_max=0.1),
            "VlabelerEditRatio100-5000ms": VlabelerEditRatio(move_min=0.1, move_max=5.0)
        }

        for pred_file in tqdm.tqdm(iterable, desc="evaluate_compute:"):
            pred_file_name = pathlib.Path(pred_file).name
            target_file = [i for i in tg_paths if pred_file_name in pathlib.Path(i).parts]
            if not target_file:
                continue
            target_file = target_file[0]

            pred_tier = label.textgrid_from_file(pred_file)[-1]
            target_tier = label.textgrid_from_file(target_file)[-1]
            pred_tier = remove_ignored_phonemes([""], pred_tier)
            target_tier = remove_ignored_phonemes([""], target_tier)

            for metric in metrics.values():
                metric.update(pred_tier, target_tier)

        result = {key: metric.compute() for key, metric in metrics.items()}

        vlabeler_loss = result["VlabelerEditRatio10-20ms"] * 0.1 + result["VlabelerEditRatio20-50ms"] * 0.2 + \
                        result["VlabelerEditRatio50-100ms"] * 0.3 + result["VlabelerEditRatio100-5000ms"] * 0.4
        result["vlabeler_loss"] = vlabeler_loss
        result["total"] = vlabeler_loss * 0.5 + result["BoundaryEditRatioWeighted"] * 0.5

        if trainer.logger:
            for metric_name, metric_value in result.items():
                trainer.model.log(f"evaluate/{metric_name}", metric_value)
