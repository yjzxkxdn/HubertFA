import os
import pathlib

import librosa
import lightning as pl
import torch
import yaml
import tqdm

from torch.utils.data import DataLoader
from einops import repeat

from networks.utils.get_melspec import MelSpecExtractor
from networks.utils.load_wav import load_wav

import networks.g2p
from networks.utils.dataset import MixedDataset, WeightedBinningAudioBatchSampler, collate_fn
from networks.task.forced_alignment import LitForcedAlignmentTask

from lightning.pytorch.callbacks import Callback, ModelCheckpoint

from networks.utils.export_tool import Exporter
from networks.utils.post_processing import post_processing

from typing import Dict

import click

from networks.utils import label
from networks.utils.metrics import Metric, VlabelerEditRatio
from evaluate import remove_ignored_phonemes


class RecentCheckpointsCallback(Callback):
    def __init__(self, dirpath, save_top_k=5, save_every_steps=5000, filename="checkpoint-{step}"):
        self.dirpath = dirpath
        self.save_top_k = save_top_k
        self.filename = filename
        self.saved_checkpoints = []
        self.save_every_steps = save_every_steps

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if trainer.global_step % self.save_every_steps == 0:
            checkpoint_path = os.path.join(
                self.dirpath,
                self.filename.format(step=trainer.global_step) + ".ckpt"
            )
            trainer.save_checkpoint(checkpoint_path)
            self.saved_checkpoints.append(checkpoint_path)

            if len(self.saved_checkpoints) > self.save_top_k:
                oldest_checkpoint = self.saved_checkpoints.pop(0)
                if os.path.exists(oldest_checkpoint):
                    os.remove(oldest_checkpoint)


class VlabelerEvaluateCallback(Callback):
    def __init__(self, evaluate_folder, dictionary, out_tg_dir, evaluate_every_steps=2000):
        super().__init__()
        self.evaluate_folder = pathlib.Path(evaluate_folder)
        self.out_tg_dir = pathlib.Path(out_tg_dir)
        self.evaluate_every_steps = evaluate_every_steps
        self.grapheme_to_phoneme = networks.g2p.DictionaryG2P(**{"dictionary": dictionary})
        self.grapheme_to_phoneme.set_in_format('lab')
        self.dataset = self.grapheme_to_phoneme.get_dataset(pathlib.Path(evaluate_folder).rglob("*.wav"))
        self.get_melspec = None

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if trainer.sanity_checking:
            return

        if self.get_melspec is None:
            self.get_melspec = MelSpecExtractor(**trainer.model.melspec_config)

        if trainer.global_step % self.evaluate_every_steps == 0:
            predictions = []
            for batch in tqdm.tqdm(self.dataset, desc="evaluate_forward:"):
                wav_path, ph_seq, word_seq, ph_idx_to_word_idx = batch
                waveform = load_wav(
                    wav_path, trainer.model.device, trainer.model.melspec_config["sample_rate"]
                )
                wav_length = waveform.shape[0] / trainer.model.melspec_config["sample_rate"]
                melspec = self.get_melspec(waveform).detach().unsqueeze(0)
                melspec = (melspec - melspec.mean()) / melspec.std()
                melspec = repeat(
                    melspec, "B C T -> B C (T N)", N=trainer.model.melspec_config["scale_factor"]
                )

                # load audio
                audio, _ = librosa.load(wav_path, sr=trainer.model.melspec_config["sample_rate"])
                if len(audio.shape) > 1:
                    audio = librosa.to_mono(audio)
                audio_t = torch.from_numpy(audio).float().to(trainer.model.device)
                audio_t = audio_t.unsqueeze(0)
                units = trainer.model.unitsEncoder.encode(audio_t, trainer.model.melspec_config["sample_rate"],
                                                          trainer.model.melspec_config["hop_length"])
                units = units.transpose(1, 2)

                units = (units - units.mean()) / units.std()
                units = repeat(
                    units, "B C T -> B C (T N)", N=trainer.model.melspec_config["scale_factor"]
                )

                if trainer.model.combine_mel:
                    input_feature = torch.cat([units, melspec], dim=1)  # [1, hubert + n_mels, T]
                else:
                    input_feature = units

                (
                    ph_seq,
                    ph_intervals,
                    word_seq,
                    word_intervals,
                    confidence,
                    _,
                    _,
                ) = trainer.model._infer_once(
                    input_feature, melspec, wav_length, ph_seq, word_seq, ph_idx_to_word_idx, False, False
                )

                predictions.append((wav_path, wav_length, confidence, ph_seq, ph_intervals, word_seq, word_intervals,))

            predictions, log = post_processing(predictions)
            out_tg_dir = self.out_tg_dir / "evaluate" / str(trainer.global_step)
            exporter = Exporter(predictions, log, out_tg_dir)
            exporter.export(['textgrid'])

            iterable = out_tg_dir.rglob("*.TextGrid")

            metrics: Dict[str, Metric] = {
                "10-20ms": VlabelerEditRatio(move_min=0.01, move_max=0.02),
                "20-50ms": VlabelerEditRatio(move_min=0.02, move_max=0.05),
                "50-100ms": VlabelerEditRatio(move_min=0.05, move_max=0.1),
                "100-5000ms": VlabelerEditRatio(move_min=0.1, move_max=5.0)
            }

            for pred_file in tqdm.tqdm(iterable, desc="evaluate_compute:"):
                target_file = list(self.evaluate_folder.rglob(pathlib.Path(pred_file).name))
                if not target_file:
                    continue
                target_file = target_file[0]

                pred_tier = label.textgrid_from_file(pred_file)[-1]
                target_tier = label.textgrid_from_file(target_file)[-1]
                pred_tier = remove_ignored_phonemes("", pred_tier)
                target_tier = remove_ignored_phonemes("", target_tier)

                for metric in metrics.values():
                    metric.update(pred_tier, target_tier)

            result = {key: metric.compute() for key, metric in metrics.items()}

            total = result["10-20ms"] * 0.1 + result["20-50ms"] * 0.2 + result["50-100ms"] * 0.3 + result[
                "100-5000ms"] * 0.4
            result["total"] = total

            if trainer.logger:
                for metric_name, metric_value in result.items():
                    trainer.logger.log_metrics(
                        {f"VlabelerEditRatio/{metric_name}": metric_value},
                        step=trainer.global_step
                    )


@click.command()
@click.option(
    "--config_path",
    "-c",
    type=str,
    default="configs/train_config.yaml",
    show_default=True,
    help="training config path",
)
@click.option(
    "--pretrained_model_path",
    "-p",
    type=str,
    default=None,
    show_default=True,
    help="pretrained model path. if None, training from scratch",
)
@click.option(
    "--resume",
    "-r",
    is_flag=True,
    default=False,
    show_default=True,
    help="resume training from checkpoint",
)
def main(config_path: str, pretrained_model_path, resume):
    os.environ[
        "TORCH_CUDNN_V8_API_ENABLED"
    ] = "1"  # Prevent unacceptable slowdowns when using 16 precision

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    with open(pathlib.Path(config["binary_folder"]) / "vocab.yaml") as f:
        vocab = yaml.safe_load(f)
    vocab_text = yaml.safe_dump(vocab)

    with open(pathlib.Path(config["binary_folder"]) / "vowel.yaml") as f:
        vowel = yaml.safe_load(f)
    vowel_text = yaml.safe_dump(vowel)

    with open(pathlib.Path(config["binary_folder"]) / "global_config.yaml") as f:
        config_global = yaml.safe_load(f)
    config.update(config_global)

    config["data_augmentation_size"] = 0

    torch.set_float32_matmul_precision(config["float32_matmul_precision"])
    pl.seed_everything(config["random_seed"], workers=True)

    # define dataset
    num_workers = config['dataloader_workers']
    train_dataset = MixedDataset(
        config["data_augmentation_size"], config["binary_folder"], prefix="train"
    )
    train_sampler = WeightedBinningAudioBatchSampler(
        train_dataset.get_label_types(),
        train_dataset.get_wav_lengths(),
        config["oversampling_weights"],
        config["batch_max_length"] / (2 if config["data_augmentation_size"] > 0 else 1),
        config["binning_length"],
        config["drop_last"],
    )
    train_dataloader = DataLoader(
        dataset=train_dataset,
        batch_sampler=train_sampler,
        collate_fn=collate_fn,
        num_workers=num_workers,
        persistent_workers=num_workers > 0,
        pin_memory=True,
        prefetch_factor=(2 if num_workers > 0 else None),
    )

    valid_dataset = MixedDataset(0, config["binary_folder"], prefix="valid")
    valid_dataloader = DataLoader(
        dataset=valid_dataset,
        batch_size=1,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=num_workers,
        persistent_workers=num_workers > 0,
    )

    # model
    lightning_alignment_model = LitForcedAlignmentTask(
        vocab_text,
        vowel_text,
        config["model"],
        config["hubert_config"],
        config["melspec_config"],
        config["optimizer_config"],
        config["loss_config"]
    )

    recent_checkpoints_callback = RecentCheckpointsCallback(
        dirpath=str(pathlib.Path("ckpt") / config["model_name"]),
        save_top_k=config["save_top_k"],
        save_every_steps=config["save_every_steps"],
        filename="checkpoint-{step}",
    )

    evaluate_folder = pathlib.Path(config["evaluate_folder"])

    vlabeler_callback = VlabelerEvaluateCallback(evaluate_folder=evaluate_folder,
                                                 dictionary=config["evaluate_dictionary"],
                                                 out_tg_dir=str(pathlib.Path("ckpt") / config["model_name"]),
                                                 evaluate_every_steps=config["evaluate_every_steps"])

    # model_checkpoint = ModelCheckpoint(
    #     dirpath=str(pathlib.Path("ckpt") / config["model_name"]),
    #     monitor="VlabelerEditRatio/total",
    #     mode="min",
    #     save_top_k=3,
    #     filename="best-{step}-{VlabelerEditRatio/total:.2f}",
    # )

    # trainer
    trainer = pl.Trainer(
        accelerator=config["accelerator"],
        devices=config["devices"],
        precision=config["precision"],
        gradient_clip_val=config["gradient_clip_val"],
        gradient_clip_algorithm=config["gradient_clip_algorithm"],
        default_root_dir=str(pathlib.Path("ckpt") / config["model_name"]),
        val_check_interval=config["val_check_interval"],
        check_val_every_n_epoch=None,
        max_epochs=-1,
        max_steps=config["optimizer_config"]["total_steps"],
        callbacks=[recent_checkpoints_callback, vlabeler_callback],
    )

    ckpt_path = None
    if pretrained_model_path is not None:
        # use pretrained model TODO: load pretrained model
        pretrained = LitForcedAlignmentTask.load_from_checkpoint(pretrained_model_path)
        lightning_alignment_model.load_pretrained(pretrained)
    elif resume:
        # resume training state
        ckpt_path_list = (pathlib.Path("ckpt") / config["model_name"]).rglob("*.ckpt")
        ckpt_path_list = sorted(
            ckpt_path_list, key=lambda x: int(x.stem.split("step=")[-1]), reverse=True
        )
        ckpt_path = str(ckpt_path_list[0]) if len(ckpt_path_list) > 0 else None

    # start training
    trainer.fit(
        model=lightning_alignment_model,
        train_dataloaders=train_dataloader,
        val_dataloaders=valid_dataloader,
        ckpt_path=ckpt_path,
    )

    # Discard the optimizer and save
    trainer.save_checkpoint(
        str(pathlib.Path("ckpt") / config["model_name"]) + ".ckpt", weights_only=True
    )


if __name__ == "__main__":
    main()
