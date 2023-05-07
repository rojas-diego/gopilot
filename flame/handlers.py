import datetime
import logging
import os
import time
from typing import List

import torch

from .tasks import LearningTask
from .trackers import NeptuneTracker, NoopTracker
from .trainer import TrainingConfig
from .utils import Metric


class TrackingHandler:
    def __init__(self, tracker: NeptuneTracker | NoopTracker):
        self.tracker = tracker

    def on_epoch_end(self, epoch_idx: int, metrics: List[Metric]):
        self.tracker.track_values([Metric("epoch", epoch_idx)])
        metrics = [Metric(f"validation/epoch/{metric.name}", metric.value) for metric in metrics]
        self.tracker.track_metrics(metrics)

    def on_test_end(self, metrics: List[Metric]):
        metrics = [Metric(f"test/{metric.name}", metric.value) for metric in metrics]
        self.tracker.track_metrics(metrics)

    def on_train_batch_end(self, epoch_idx: int, batch_idx: int, metrics: List[Metric]):
        metrics = [Metric(f"train/batch/{metric.name}", metric.value) for metric in metrics]
        self.tracker.track_metrics(metrics)

    def on_validation_batch_end(self, epoch_idx: int, batch_idx: int, metrics: List[Metric]):
        metrics = [Metric(f"validation/batch/{metric.name}", metric.value) for metric in metrics]
        self.tracker.track_metrics(metrics)

    def on_test_batch_end(self, batch_idx: int, metrics: List[Metric]):
        metrics = [Metric(f"test/batch/{metric.name}", metric.value) for metric in metrics]
        self.tracker.track_metrics(metrics)


class LoggingHandler:
    """
    Log the run's hyperparameters and the validation accuracy and loss at the
    end of each epoch.
    """

    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        self.epoch_start_ts = time.time()

    def on_train_start(self, device: str | torch.device, config: TrainingConfig):
        logging.info(f"Starting training | Accelerator: {device} | Epochs: {config.num_epochs} | Gradient accumulation: {config.gradient_accumulation}")

    def on_epoch_start(self, epoch_idx: int):
        self.epoch_start_ts = time.time()

    def on_epoch_end(self, epoch_idx: int, metrics: List[Metric]):
        elapsed = time.time() - self.epoch_start_ts # type: ignore
        elapsed_str = str(datetime.timedelta(seconds=elapsed))
        metrics_string = self._metrics_to_string("Validation", metrics)
        logging.info(f"Completed Epoch {epoch_idx+1:3} | Elapsed: {elapsed_str} | {metrics_string}")

    def on_test_end(self, metrics: List[Metric]):
        metrics_string = self._metrics_to_string("Test", metrics)
        logging.info(f"Completed Test | {metrics_string}")

    def on_step(self, epoch_idx: int, batch_idx: int, step_idx: int, metrics: List[Metric]):
        if self.verbose:
            metrics_string = self._metrics_to_string(None, metrics)
            logging.info(f"Epoch {epoch_idx+1:3} | Step {step_idx+1:5} | {metrics_string}")

    def _metrics_to_string(self, prefix: str | None, metrics: List[Metric]) -> str:
        if prefix is None:
            return ", ".join([f"{metric.name.capitalize()}: {metric.value:.4f}" for metric in metrics])
        return ", ".join([f"{prefix} {metric.name.capitalize()}: {metric.value:.4f}" for metric in metrics])


class CheckpointingHandler:
    def __init__(self, checkpoints_dir: str, filename_prefix: str, max_epoch_interval: int = 1, max_step_interval: int = 120, max_time_interval_sec: int = 60*2):
        self.checkpoints_dir = checkpoints_dir
        self.filename_prefix = filename_prefix
        self.max_step_interval = max_step_interval
        self.max_epoch_interval = max_epoch_interval
        self.max_time_interval_sec = max_time_interval_sec
        self.files_written = []
        self.last_checkpoint_ts = time.time()
        self.last_checkpoint_epoch = 0
        self.last_checkpoint_step = 0
        # Ensure the checkpoints directory exists
        os.makedirs(self.checkpoints_dir, exist_ok=True)

    def checkpoint(self, task: LearningTask, epoch_idx: int, batch_idx: int, step_idx: int, metrics: List[Metric]):
        if self._should_checkpoint(epoch_idx, batch_idx, step_idx):
            metrics_dict = {metric.name: metric.value for metric in metrics}
            metrics_dict["epoch"] = epoch_idx
            metrics_dict["step"] = step_idx
            metrics_dict["batch"] = batch_idx
            formatted_filename_prefix = self.filename_prefix.format(**metrics_dict)
            # Check if formatted filename still contain format markers
            if "{" in formatted_filename_prefix:
                logging.warning(f"Checkpoint filename {formatted_filename_prefix} still contains format markers. Checkpointing will be skipped.")
                return
            location = os.path.join(self.checkpoints_dir, formatted_filename_prefix)
            self.last_checkpoint_ts = time.time()
            self.last_checkpoint_epoch = epoch_idx
            self.last_checkpoint_step = step_idx
            task.checkpoint(location, epoch_idx, batch_idx, step_idx, metrics)
            # Cleanup old checkpoints
            for file in self.files_written:
                os.remove(file)
            self.files_written = [location]
            logging.info(f"Saved model to '{location}'")

    def _should_checkpoint(self, epoch_idx: int, batch_idx: int, step_idx: int) -> bool:
        """Only checkpoint if at least one interval has passed since the last checkpoint."""
        if epoch_idx - self.last_checkpoint_epoch >= self.max_epoch_interval:
            return True
        if step_idx - self.last_checkpoint_step >= self.max_step_interval:
            return True
        if time.time() - self.last_checkpoint_ts >= self.max_time_interval_sec:
            return True
        return False
