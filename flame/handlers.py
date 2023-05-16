import datetime
import logging
import os
import time
from typing import List, Optional, Union

from torch.profiler import profile

from .trackers import NeptuneTracker, NoopTracker
from .utils import Metric
from .trainer import Trainer


class TrackingHandler:
    def __init__(self, tracker: Union[NeptuneTracker, NoopTracker], on_step: bool = True, on_batch: bool = True):
        self._on_step = on_step
        self._on_batch = on_batch
        self.tracker = tracker

    def on_step(self, trainer: Trainer, epoch_idx: int, batch_idx: int, step_idx: int, metrics: List[Metric]):
        if self._on_step:
            metrics = [Metric(f"train/step/{metric.name}", metric.value, step=metric.step) for metric in metrics]
            self.tracker.track_metrics(metrics)
        self.tracker.track_values([Metric("train/steps", step_idx)])

    def on_train_batch_end(self, trainer: Trainer, epoch_idx: int, batch_idx: int, step_idx: int, metrics: List[Metric]):
        if self._on_batch:
            metrics = [Metric(f"train/batch/{metric.name}", metric.value, step=metric.step) for metric in metrics]
            self.tracker.track_metrics(metrics)
        self.tracker.track_values([Metric("train/batches", batch_idx)])

    def on_epoch_end(self, trainer: Trainer, epoch_idx: int):
        self.tracker.track_values([Metric("train/epochs", epoch_idx)])

    def on_validation_end(self, trainer: Trainer, epoch_idx: int, metrics: List[Metric]):
        metrics = [Metric(f"validation/{metric.name}", metric.value, step=metric.step) for metric in metrics]
        self.tracker.track_values(metrics)

    def on_validation_batch_end(self, trainer: Trainer, epoch_idx: int, batch_idx: int, metrics: List[Metric]):
        if self._on_batch:
            metrics = [Metric(f"validation/batch/{metric.name}", metric.value, step=metric.step) for metric in metrics]
            self.tracker.track_metrics(metrics)

    def on_test_batch_end(self, trainer: Trainer, batch_idx: int, metrics: List[Metric]):
        metrics = [Metric(f"test/batch/{metric.name}", metric.value, step=metric.step) for metric in metrics]
        self.tracker.track_metrics(metrics)

    def on_test_end(self, trainer: Trainer, metrics: List[Metric]):
        metrics = [Metric(f"test/{metric.name}", metric.value, step=metric.step) for metric in metrics]
        self.tracker.track_values(metrics)


class LoggingHandler:
    """
    Log the run's hyperparameters and the validation accuracy and loss at the
    end of each epoch.
    """

    def __init__(self, on_step: bool = True, on_batch: bool = True):
        self._on_step = on_step
        self._on_batch = on_batch
        self.epoch_start_ts = time.time()
        self.last_step_ts = time.time()

    def on_train_start(self, trainer: Trainer):
        logging.info(f"Starting training | Accelerator: {trainer.device} | Epochs: {trainer.num_epochs}")
        self.last_step_ts = time.time()

    def on_epoch_start(self, trainer: Trainer, epoch_idx: int):
        self.epoch_start_ts = time.time()

    def on_validation_end(self, trainer: Trainer, epoch_idx: int, metrics: List[Metric]):
        elapsed = time.time() - self.epoch_start_ts
        elapsed_str = str(datetime.timedelta(seconds=elapsed))
        metrics_string = self._metrics_to_string("Validation", metrics)
        logging.info(f"Completed Epoch {epoch_idx+1:3} | Elapsed: {elapsed_str} | {metrics_string}")

    def on_test_end(self, trainer: Trainer, metrics: List[Metric]):
        metrics_string = self._metrics_to_string("Test", metrics)
        logging.info(f"Completed Test | {metrics_string}")

    def on_step(self, trainer: Trainer, epoch_idx: int, batch_idx: int, step_idx: int, metrics: List[Metric]):
        if self._on_step:
            metrics_string = self._metrics_to_string(None, metrics)
            elapsed_ms_str = f"{(time.time() - self.last_step_ts) * 1000:.2f}ms"
            self.last_step_ts = time.time()
            logging.info(f"Epoch {epoch_idx+1:3} | Step {step_idx+1:5} ({elapsed_ms_str}) | {metrics_string}")

    def on_train_batch_end(self, trainer: Trainer, epoch_idx: int, batch_idx: int, step_idx: int, metrics: List[Metric]):
        if self._on_batch:
            metrics_string = self._metrics_to_string(None, metrics)
            logging.info(f"Epoch {epoch_idx+1:3} | Batch {batch_idx+1:5} | {metrics_string}")

    def _metrics_to_string(self, prefix: Optional[str], metrics: List[Metric]) -> str:
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
        logging.info(f"Checkpointing enabled. Checkpoints will be saved to {self.checkpoints_dir}")

    def on_step(self, trainer: Trainer, epoch_idx: int, batch_idx: int, step_idx: int, metrics: List[Metric]):
        if self._should_checkpoint(epoch_idx, batch_idx, step_idx):
            formatted_filename_prefix = self._make_filename_prefix(epoch_idx, batch_idx, step_idx, metrics)
            # Check if formatted filename still contain format markers
            if "{" in formatted_filename_prefix:
                logging.warning(f"Checkpoint filename {formatted_filename_prefix} still contains format markers. Checkpointing will be skipped.")
                return
            self._checkpoint(os.path.join(self.checkpoints_dir, formatted_filename_prefix), trainer, epoch_idx, batch_idx, step_idx, metrics)

    def _checkpoint(self, location: str, trainer: Trainer, epoch_idx: int, batch_idx: int, step_idx: int, metrics: List[Metric]):
        self.last_checkpoint_ts = time.time()
        self.last_checkpoint_epoch = epoch_idx
        self.last_checkpoint_step = step_idx
        trainer.task.checkpoint(location, epoch_idx, batch_idx, step_idx, metrics)
        # Cleanup old checkpoints
        for file in self.files_written:
            if os.path.exists(file):
                os.remove(file)
        self.files_written = [location]
        logging.info(f"Saved model to '{location}'")

    def _make_filename_prefix(self, epoch_idx: int, batch_idx: int, step_idx: int, metrics: List[Metric]) -> str:
        metrics_dict = {metric.name: metric.value for metric in metrics}
        metrics_dict["epoch"] = epoch_idx
        metrics_dict["step"] = step_idx
        metrics_dict["batch"] = batch_idx
        return self.filename_prefix.format(**metrics_dict)

    def _should_checkpoint(self, epoch_idx: int, batch_idx: int, step_idx: int) -> bool:
        """Only checkpoint if at least one interval has passed since the last checkpoint."""
        if epoch_idx - self.last_checkpoint_epoch >= self.max_epoch_interval:
            return True
        if step_idx - self.last_checkpoint_step >= self.max_step_interval:
            return True
        if time.time() - self.last_checkpoint_ts >= self.max_time_interval_sec:
            return True
        return False


class TorchProfilingHandler:
    def __init__(self, output_filename: str, *args, **kwargs):
        self.output_filename = output_filename
        self.profiler_args = args
        self.profiler_kwargs = kwargs
        self.profiler = None

    def __enter__(self):
        if self.profiler is None:
            self.profiler = profile(*self.profiler_args, on_trace_ready=self._on_trace_ready, **self.profiler_kwargs)
        self.profiler.__enter__()
        logging.info(f"Profiling enabled. Results will be saved to '{self.output_filename}'")
        return self

    def __exit__(self, *args):
        logging.info(f"Shutting down profiling.")
        if self.profiler:
            self.profiler.__exit__(*args)
            self.profiler = None

    def _on_trace_ready(self, profiler):
        output = profiler.key_averages().table(row_limit=10, sort_by="self_cpu_time_total")
        profiler.export_chrome_trace(self.output_filename)
        logging.info(f"Profiling available at '{self.output_filename}'")
        print(output)

    def on_train_start(self, trainer: Trainer):
        self.__enter__()

    def on_train_end(self, trainer: Trainer):
        self.__exit__(None, None, None)

    def on_step(self, trainer: Trainer, epoch_idx: int, batch_idx: int, step_idx: int, metrics: List[Metric]):
        if self.profiler:
            self.profiler.step()

    def cleanup(self, trainer: Trainer):
        self.__exit__(None, None, None)
