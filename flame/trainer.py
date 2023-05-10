import atexit
import logging
import signal
import sys
from typing import Iterable, List

import torch
import tqdm

from .tasks import LearningTask
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
        task: LearningTask,
        device: str | torch.device,
        handlers: List = [],
    ):
        self.task = task
        self.device = torch.device(device)
        self.handlers = handlers
        self.is_training = False
        self.is_testing = False

    def train(self, train_loader: Iterable, validation_loader: Iterable | None = None, num_epochs: int = 1, gradient_accumulation_steps: int = 1, enable_progress_bar: bool = False):
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
            samples = self._make_progress_bar(True, "Test", self.test_loader)
            for batch_idx, batch in samples:
                self._callback("on_test_batch_start", batch_idx)
                metrics = self._try_forward(batch, False, -1, batch_idx, -1)
                self._callback("on_test_batch_end", batch_idx, metrics)
                metrics_store.accumulate(metrics)
                samples.update(metrics)
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
        samples = self._make_progress_bar(self.enable_progress_bar, f"Train Epoch {epoch_idx+1}", self.train_loader)
        step_idx = 0
        for batch_idx, batch in samples:
            self._callback("on_train_batch_start", epoch_idx, batch_idx, step_idx)
            batch_metrics = self._try_forward(batch, True, epoch_idx, batch_idx, step_idx)
            if (batch_idx + 1) % self.gradient_accumulation_steps == 0:
                step_metrics = self._try_step(epoch_idx, batch_idx, step_idx)
                self._callback("on_step", epoch_idx, batch_idx, step_idx, step_metrics)
                samples.update(step_metrics)
                step_idx += 1
            self._callback("on_train_batch_end", epoch_idx, batch_idx, step_idx, batch_metrics)

    def _validate(self, epoch_idx: int):
        if self.validation_loader is None:
            return None
        metrics_store = MetricsStore()
        self.task.eval(self.device)
        self._callback("on_validation_start", epoch_idx)
        with torch.no_grad():
            samples = self._make_progress_bar(self.enable_progress_bar, f"Validate Epoch {epoch_idx+1}", self.validation_loader)
            for batch_idx, batch in samples:
                self._callback("on_validation_batch_start", epoch_idx, batch_idx)
                metrics = self._try_forward(batch, False, epoch_idx, batch_idx, -1)
                self._callback("on_validation_batch_end", epoch_idx, batch_idx, metrics)
                metrics_store.accumulate(metrics)
                samples.update(metrics)
            self._callback("on_validation_start", epoch_idx, metrics_store.mean())

    def _try_forward(self, batch, backprop: bool, epoch_idx, step_idx: int, batch_idx):
        try:
            return self.task.forward(batch, self.device, backprop=backprop)
        except Exception as e:
            logging.error(f"Exception raised during {self.task.__class__.__name__}.forward() (epoch={epoch_idx+1}, batch={batch_idx}, step={step_idx}). Attempting graceful exit.")
            self._exception()
            raise e
        
    def _try_step(self, epoch_idx, batch_idx, step_idx: int):
        try:
            return self.task.step(self.device)
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
                getattr(handler, callback_name)(self, *args, **kwargs)

    def _make_progress_bar(self, enable_progress_bar: bool, desc: str, loader: Iterable):
        base = enumerate(loader)
        if enable_progress_bar:
            if sys.stdout.isatty():
                if hasattr(loader, 'dataset'):
                    if hasattr(loader.dataset, '__len__'): # type: ignore
                        return TQDMProgressBar(tqdm.tqdm(base, desc=desc, leave=False, mininterval=0.1, total=len(loader.dataset))) # type: ignore
                return TQDMProgressBar(tqdm.tqdm(base, desc=desc, leave=False, mininterval=0.1))
            else:
                logging.warning("Progress bar is not supported in non-interactive mode.")
        return NoopProgressBar(base)


class NoopProgressBar:
    def __init__(self, iterable):
        self.iterable = iterable

    def __iter__(self):
        return iter(self.iterable)

    def update(self, metrics: List[Metric]):
        pass


class TQDMProgressBar:
    def __init__(self, iterable):
        self.iterable = iterable

    def __iter__(self):
        return iter(self.iterable)

    def update(self, metrics: List[Metric]):
        self.iterable.set_postfix_str(", ".join(f"{metric.name.capitalize()}: {metric.value:.4f}" for metric in metrics), refresh=False)
