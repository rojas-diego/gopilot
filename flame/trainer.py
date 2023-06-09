import atexit
import logging
import signal
import sys
from typing import Iterable, List, Optional, Union

import torch
import tqdm

from .tasks import Task
from .utils import Metric, MetricsStore


class Trainer:
    """
    A trainer that uses handlers to customize the training process.

    By default, the trainer will handle training, validation and testing.
    The handlers can be used to add custom functionality to the training
    process such as logging, checkpointing, etc.

    Handlers callbacks:
    - on_train_start
    - on_epoch_start
    - on_train_batch_start
    - on_step
    - on_train_batch_end
    - on_validation_start
    - on_validation_batch_start
    - on_validation_batch_end
    - on_validation_end
    - on_epoch_end
    - on_train_end
    - on_test_start
    - on_test_end

    Args:
        task (LearningTask): The learning task to train.
        device (str | torch.device): The device to use for training.
        train_loader (Iterable): The data loader for the training set.
        validation_loader (Iterable, optional): The data loader for the validation set.
        test_loader (Iterable, optional): The data loader for the test set.
        handlers (List, optional): A list of handlers to use.

    Examples:
        >>> import flame
        ... trainer = Trainer(
        ...     task=flame.MultiClassSupervisedLearningTask(model, criterion, optimizer),
        ...     device=flame.best_device(),
        ...     handlers=[LogginHandler(), TrackingHandler()]
        ... )
        ... trainer.train(train_loader, num_epochs=10)
    """

    def __init__(
        self,
        task: Task,
        device: Union[str, torch.device],
        handlers: List = [],
    ):
        self.task = task
        self.device = torch.device(device)
        self.handlers = handlers
        self.is_training = False
        self.is_testing = False

    def train(self, train_loader: Iterable, validation_loader: Optional[Iterable] = None, num_epochs: int = 1, gradient_accumulation_steps: int = 1, enable_progress_bar: bool = False):
        self.num_epochs = num_epochs
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.enable_progress_bar = enable_progress_bar
        self.validation_loader = validation_loader
        self.train_loader = train_loader
        self._setup_training()
        self._callback("on_train_start")
        for epoch_idx in range(num_epochs if num_epochs > 0 else sys.maxsize):
            self._callback("on_epoch_start", epoch_idx)
            self._train_epoch(epoch_idx)
            self._validate(epoch_idx)
            self._callback("on_epoch_end", epoch_idx)
        self._callback("on_train_end")
        self._teardown_training()

    def test(self, test_loader: Iterable):
        self.test_loader = test_loader
        self._setup_testing()
        self._callback("on_test_start")
        metrics_store = MetricsStore()
        with torch.no_grad():
            samples = enumerate(self.test_loader)
            for batch_idx, batch in samples:
                self._callback("on_test_batch_start", batch_idx)
                metrics = self._try_forward(batch, False, -1, batch_idx, -1)
                self._callback("on_test_batch_end", batch_idx, metrics)
                metrics_store.accumulate(metrics)
        metrics = metrics_store.mean()
        self._callback("on_test_end", metrics)
        return metrics

    def predict(self, batch):
        self.task.eval(self.device)
        with torch.no_grad():
            return self._try_forward(batch, False, -1, -1, -1)

    def register_handlers(self, *handlers):
        self.handlers.extend(handlers)

    def _setup_training(self):
        self.is_training = True
        self.task.train(self.device)
        self._register_signal_and_exit_handlers()

    def _setup_testing(self):
        self.is_testing = True
        self.task.eval(self.device)
        self._register_signal_and_exit_handlers()

    def _train_epoch(self, epoch_idx: int):
        self.task.train(self.device)
        step_idx = 0
        samples = enumerate(self.train_loader)
        # Prefetch each batch to avoid blocking the GPU
        batch_idx, batch = next(samples, (None, None))
        if batch_idx is not None:
            batch = batch.to(self.device, non_blocking=True) # type: ignore
        while batch_idx is not None:
            next_batch_idx, next_batch = next(samples, (None, None))
            if next_batch_idx is not None:
                next_batch = next_batch.to(self.device, non_blocking=True) # type: ignore
            self._callback("on_train_batch_start", epoch_idx, batch_idx, step_idx)
            batch_metrics = self._try_forward(batch, True, epoch_idx, batch_idx, step_idx)
            if (batch_idx + 1) % self.gradient_accumulation_steps == 0:
                step_metrics = self._try_step(batch, epoch_idx, batch_idx, step_idx)
                self._callback("on_step", epoch_idx, batch_idx, step_idx, step_metrics)
                step_idx += 1
                self._invoke_checkpointing_callbacks(epoch_idx, batch_idx, step_idx, step_metrics)
            self._callback("on_train_batch_end", epoch_idx, batch_idx, step_idx, batch_metrics)
            batch_idx, batch = next_batch_idx, next_batch

    def _invoke_checkpointing_callbacks(self, epoch_idx: int, batch_idx: int, step_idx: int, step_metrics: List[Metric]):
        for handler in self.handlers:
            if hasattr(handler, "should_checkpoint") and hasattr(handler, "checkpoint"):
                try:
                    if getattr(handler, "should_checkpoint")(self, epoch_idx, batch_idx, step_idx, step_metrics):
                        checkpoint_filepath = getattr(handler, "checkpoint")(self, epoch_idx, batch_idx, step_idx, step_metrics)
                        self._callback("on_checkpoint", epoch_idx, batch_idx, step_idx, checkpoint_filepath)
                except Exception as e:
                    logging.error(f"Exception raised during {handler.__class__.__name__}.should_checkpoint() or {handler.__class__.__name__}.checkpoint(): {e}.")

    def _should_checkpoint(self, epoch_idx: int, batch_idx: int, step_idx: int, step_metrics: List[Metric]):
        for handler in self.handlers:
            if hasattr(handler, "should_checkpoint"):
                try:
                    if getattr(handler, "should_checkpoint")(self, epoch_idx, batch_idx, step_idx, step_metrics):
                        return True
                except Exception as e:
                    logging.error(f"Exception raised during {handler.__class__.__name__}.should_checkpoint(): {e}.")

    def _validate(self, epoch_idx: int):
        if self.validation_loader is None:
            return None
        metrics_store = MetricsStore()
        self.task.eval(self.device)
        self._callback("on_validation_start", epoch_idx)
        with torch.no_grad():
            for batch_idx, batch in enumerate(self.validation_loader):
                self._callback("on_validation_batch_start", epoch_idx, batch_idx)
                metrics = self._try_forward(batch, False, epoch_idx, batch_idx, -1)
                self._callback("on_validation_batch_end", epoch_idx, batch_idx, metrics)
                metrics_store.accumulate(metrics)
            self._callback("on_validation_end", epoch_idx, metrics_store.mean())

    def _try_forward(self, batch, backprop: bool, epoch_idx, step_idx: int, batch_idx):
        try:
            return self.task.forward(batch, self.device, backprop=backprop)
        except Exception as e:
            logging.error(f"Exception raised during {self.task.__class__.__name__}.forward() (epoch={epoch_idx+1}, batch={batch_idx}, step={step_idx}). Attempting graceful exit.")
            self._exception()
            raise e

    def _try_step(self, batch, epoch_idx: int, batch_idx: int, step_idx: int):
        try:
            return self.task.step(batch, self.device, epoch_idx, batch_idx, step_idx)
        except Exception as e:
            logging.error(f"Exception raised during {self.task.__class__.__name__}.step() (epoch={epoch_idx+1}, batch={batch_idx}, step={step_idx}). Attempting graceful exit.")
            self._exception()
            raise e

    def _teardown_training(self, metrics: dict = {}):
        self.is_training = False
        self._unregister_signal_and_exit_handlers()

    def _teardown_testing(self):
        self.is_testing = False
        self._unregister_signal_and_exit_handlers()

    def _register_signal_and_exit_handlers(self):
        atexit.register(self._exit)
        signal.signal(signal.SIGTERM, self._signal_handler)
        signal.signal(signal.SIGINT, self._signal_handler)

    def _unregister_signal_and_exit_handlers(self):
        atexit.unregister(self._exit)
        signal.signal(signal.SIGTERM, signal.SIG_DFL)
        signal.signal(signal.SIGINT, signal.SIG_DFL)

    def _exception(self):
        self._callback("cleanup")
        if self.is_training:
            self._teardown_training()
        if self.is_testing:
            self._teardown_testing()

    def _exit(self):
        logging.warning("Unexpected stop. Attempting graceful exit.")
        self._unregister_signal_and_exit_handlers()
        self._callback("cleanup")

    def _signal_handler(self, sig, frame):
        logging.warning("Received signal. Attempting graceful exit.")
        self._unregister_signal_and_exit_handlers()
        self._callback("cleanup")
        sys.exit(0)

    def _callback(self, callback_name: str, *args, **kwargs):
        for handler in self.handlers:
            if hasattr(handler, callback_name):
                try:
                    getattr(handler, callback_name)(self, *args, **kwargs)
                except Exception as e:
                    logging.error(f"Exception raised during {handler.__class__.__name__}.{callback_name}(): {e}. Ignoring.")
