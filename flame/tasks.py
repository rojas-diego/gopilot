from abc import ABC, abstractmethod
from typing import Any, List

import torch
from torch.nn import Module
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler

from flame.utils import Metric

from .utils import Metric


class Task(ABC):
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


class SimpleTask(Task):
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
