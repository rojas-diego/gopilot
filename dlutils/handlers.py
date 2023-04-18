import logging
import time
from typing import List

from dlutils.trackers import NeptuneTracker, NoopTracker
from dlutils.utils import Metric


class TrackingHandler:
    """
    Reports the run's hyperparameters and metrics to Neptune.ai.
    """

    def __init__(self, tracker: NeptuneTracker | NoopTracker):
        self.tracker = tracker

    def on_epoch_end(self, epoch_idx: int, metrics: List[Metric]):
        self.tracker.track_epoch(epoch_idx+1)
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

    def __init__(self):
        self.epoch_start_ts = None

    def on_epoch_start(self, epoch_idx: int):
        self.epoch_start_ts = time.time()

    def on_epoch_end(self, epoch_idx: int, metrics: List[Metric]):
        elapsed = time.time() - self.epoch_start_ts
        metrics_string = ", ".join([f"Validation {metric.name.capitalize()}: {metric.value:.4f}" for metric in metrics])
        logging.info(f"Completed Epoch {epoch_idx+1:3} | Elapsed: {elapsed} | {metrics_string}")

    def on_test_end(self, metrics: List[Metric]):
        metrics_string = ", ".join([f"{metric.name.capitalize()}: {metric.value:.4f}" for metric in metrics])
        logging.info(f"Completed Test | {metrics_string}")
