import math
from abc import ABC, abstractmethod
from typing import Any, List

import torch
from torch.nn import Module
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torch.nn.utils.clip_grad import clip_grad_norm_ as torch_clip_grad_norm

from flame.utils import Metric

from .utils import Metric


class LearningTask(ABC):
    @abstractmethod
    def train(self, device: torch.device):
        """Called at the start of training."""
        pass

    @abstractmethod
    def eval(self, device: torch.device):
        """Called at the start of validation and testing."""
        pass

    @abstractmethod
    def forward(self, batch: Any, device: torch.device, backprop: bool) -> List[Metric]:
        """Called during training, validation and testing for each batch of
        samples in the dataset."""
        pass

    @abstractmethod
    def step(self, device: torch.device) -> List[Metric]:
        """Called during training to perform gradient updates."""
        pass

    @abstractmethod
    def checkpoint(self, location: str, epoch_idx: int, batch_idx: int, step_idx: int, metrics: List[Metric]):
        """Called freely by handlers through the `checkpoint` callback"""
        pass


class SupervisedLearningTask(LearningTask):
    def __init__(self, model: Module, criterion: Module, optimizer: Optimizer, scheduler: LRScheduler | None = None):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler

    def train(self, device: torch.device):
        self.model.train()
        self.move_to_device(device)

    def eval(self, device: torch.device):
        self.model.eval()
        self.move_to_device(device)

    def move_to_device(self, device: torch.device):
        self.model.to(device)
        self.criterion.to(device)
        # See: https://github.com/pytorch/pytorch/issues/2830
        for state in self.optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(device)

    def checkpoint(self, location: str, epoch_idx: int, batch_idx: int, step_idx: int, metrics: List[Metric]):
        torch.save({
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": None if self.scheduler is None else self.scheduler.state_dict(),
            "epoch_idx": epoch_idx,
            "batch_idx": batch_idx,
            "step_idx": step_idx,
        }, location)


class TransformerLanguageModelingTask(SupervisedLearningTask):
    """
    Expects a single tensor of padded sequences of shape `(batch_size,
    max_sequence_len+1)`. Each sequence consist of a torch long tensor of token
    IDs.

    For all sequences in the batch, the model is trained to predict the next
    token in the sequence. Each sequence is used to make multiple predictions.

    Reports "perplexity" and "loss" metrics.
    """
    def __init__(self, model: Module, criterion: Module, optimizer: Optimizer, scheduler: LRScheduler | None = None, clip_gradients: float | None = None):
        super().__init__(model, criterion, optimizer, scheduler)
        self.clip_gradients = clip_gradients
        self.step_loss = []

    def forward(self, batch: torch.Tensor, device: torch.device, backprop: bool):
        if backprop:
            self.optimizer.zero_grad()

        batch = batch.to(device)
        batch_size, sequence_length = batch.shape[0], batch.shape[1]-1

        # Shift the batch by 1 position to create the target batch
        input_batch = batch[:, :-1]
        target_batch = batch[:, 1:]

        outputs: torch.Tensor = self.model(input_batch) # (batch_size, sequence_len, vocab_size)

        outputs = outputs.view(-1, outputs.size(-1)) # (batch_size * sequence_length, vocab_size)
        target_batch = target_batch.reshape(-1) # (batch_size * sequence_length)

        # Calculate the loss for the entire batch
        loss = self.criterion(outputs, target_batch)
        loss_value = loss.item()
        self.step_loss.append(loss_value)

        # Backward pass
        if backprop:
            loss.backward()

        return [
            Metric("loss", loss_value, weight=batch_size * sequence_length),
            Metric("perplexity", math.exp(min(loss_value, 100)), weight=batch_size * sequence_length),
        ]

    def step(self, device: torch.device) -> List[Metric]:
        if self.clip_gradients is not None:
            torch_clip_grad_norm(self.model.parameters(), self.clip_gradients)

        self.optimizer.step()
        self.optimizer.zero_grad()

        if self.scheduler is not None:
            self.scheduler.step()

        num_losses = len(self.step_loss)
        step_loss_value = sum(self.step_loss) / num_losses
        self.step_loss = []

        return [
            Metric("loss", step_loss_value, weight=num_losses),
            Metric("perplexity", math.exp(min(step_loss_value, 100)), weight=num_losses),
            Metric("lr", self.optimizer.param_groups[0]["lr"]),
        ]
