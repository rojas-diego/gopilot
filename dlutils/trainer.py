from typing import Iterable, List

import torch
import tqdm
from torch.utils.data import DataLoader, IterDataPipe

from dlutils.tasks import LearningTask
from .utils import MetricsStore


class Trainer:
    """
    A trainer that uses callbacks to handle the training process.

    By default, the trainer will handle training, validation and testing.
    The callbacks can be used to add custom functionality to the training
    process such as logging, checkpointing, etc.

    Callbacks:
    - on_train_start
    - on_train_end
    - on_epoch_start
    - on_epoch_end
    - on_train_batch_start
    - on_train_batch_end
    - on_validation_batch_start
    - on_validation_batch_end
    - on_test_start
    - on_test_end

    Args:
        task (LearningTask): The learning task to train.
        device (str): The device to use for training.
        train_loader (DataLoader): The data loader for the training set.
        validation_loader (DataLoader, optional): The data loader for the validation set.
        test_loader (DataLoader, optional): The data loader for the test set.
        handlers (List, optional): A list of callbacks to use.

    Examples:
        >>> import dlutils
        ... trainer = Trainer(
        ...     task=dlutils.MultiClassSupervisedLearningTask(model, criterion, optimizer),
        ...     device="cuda",
        ...     train_loader=train_loader,
        ...     validation_loader=validation_loader,
        ...     test_loader=test_loader,
        ...     handlers=[LogginHandler(), TrackingHandler()]
        ... )
        ... trainer.train(num_epochs=10)
    """

    def __init__(
        self,
        task: LearningTask,
        device: str | torch.device,
        train_loader: Iterable,
        validation_loader: Iterable | None,
        test_loader: Iterable | None,
        handlers: List = [],
    ):
        self.task = task
        self.device = torch.device(device)
        self.train_loader = train_loader
        self.validation_loader = validation_loader
        self.test_loader = test_loader
        self.handlers = handlers

    def train(self, num_epochs: int):
        self._callback("on_train_start")
        for epoch_idx in range(num_epochs):
            self._callback("on_epoch_start", epoch_idx)
            self._train_epoch(epoch_idx)
            metrics = self._validate_epoch(epoch_idx)
            self._callback("on_epoch_end", epoch_idx, metrics)
        self._callback("on_train_end", num_epochs)

    def test(self):
        if self.test_loader is None:
            raise ValueError("No test loader was provided.")

        self.task.eval(self.device)
        self._callback("on_test_start")
        metrics_store = MetricsStore()

        with torch.no_grad():
            samples = self._make_progress_bar("Test", self.test_loader)
            for batch_idx, batch in samples:
                self._callback("on_test_batch_start", batch_idx)
                metrics = self.task.forward(batch, self.device, backprop=False)
                metrics_store.accumulate(metrics)
                self._callback("on_test_batch_end", batch_idx, metrics)
                samples.set_postfix_str(", ".join(f"{metric.name.capitalize()}: {metric.value:.4f}" for metric in metrics), refresh=False)

        metrics = metrics_store.mean()
        self._callback("on_test_end", metrics)
        return metrics

    def register_handlers(self, *handlers):
        self.handlers.extend(handlers) # type: ignore

    def _train_epoch(self, epoch_idx: int):
        self.task.train(self.device)
        metrics_store = MetricsStore()
        samples = self._make_progress_bar(f"Train Epoch {epoch_idx+1}", self.train_loader)
        for batch_idx, batch in samples:
            self._callback("on_train_batch_start", epoch_idx, batch_idx)
            metrics = self.task.forward(batch, self.device, backprop=True)
            metrics_store.accumulate(metrics)
            self._callback("on_train_batch_end", epoch_idx, batch_idx, metrics)
            samples.set_postfix_str(", ".join(f"{metric.name.capitalize()}: {metric.value:.4f}" for metric in metrics), refresh=False)

    def _validate_epoch(self, epoch_idx: int):
        if self.validation_loader is None:
            return None

        self.task.eval(self.device)
        metrics_store = MetricsStore()

        with torch.no_grad():
            samples = self._make_progress_bar(f"Validate Epoch {epoch_idx+1}", self.validation_loader)
            for batch_idx, batch in samples:
                self._callback("on_validation_batch_start", epoch_idx, batch_idx)
                metrics = self.task.forward(batch, self.device, backprop=False)
                metrics_store.accumulate(metrics)
                self._callback("on_validation_batch_end", epoch_idx, batch_idx, metrics)
                samples.set_postfix_str(", ".join(f"{metric.name.capitalize()}: {metric.value:.4f}" for metric in metrics), refresh=False)

        return metrics_store.mean()

    def _callback(self, callback_name: str, *args, **kwargs):
        for handler in self.handlers:
            if hasattr(handler, callback_name):
                getattr(handler, callback_name)(*args, **kwargs)

    def _make_progress_bar(self, desc: str, loader: IterDataPipe | DataLoader):
        if isinstance(loader, DataLoader):
            return tqdm.tqdm(enumerate(loader), desc=desc, leave=False, mininterval=0.1, total=len(loader))
        else:
            return tqdm.tqdm(enumerate(loader), desc=desc, leave=False, mininterval=0.1)
