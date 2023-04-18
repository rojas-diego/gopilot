import math
from abc import ABC, abstractmethod
from typing import Any, List, Tuple

import torch
from torch.nn import Module
from torch.optim import Optimizer

from .utils import Metric


class LearningTask(ABC):
    @abstractmethod
    def train(self, device: torch.device) -> None:
        pass

    @abstractmethod
    def eval(self, device: torch.device) -> None:
        pass

    @abstractmethod
    def forward(self, batch: Any, device: torch.device, backprop: bool) -> List[Metric]:
        pass


class SupervisedLearningTask(LearningTask):
    def __init__(self, model: Module, criterion: Module, optimizer: Optimizer):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer

    def train(self, device: torch.device) -> None:
        self.model.train()
        self._move_to_device(device)

    def eval(self, device: torch.device) -> None:
        self.model.eval()
        self._move_to_device(device)

    def _move_to_device(self, device: torch.device):
        self.model.to(device)
        self.criterion.to(device)
        # See: https://github.com/pytorch/pytorch/issues/2830
        for state in self.optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(device)


class SupervisedClassificationTask(SupervisedLearningTask):
    """
    Expects batch to be a tuple of (inputs, targets) where inputs is a tensor of
    shape (batch_size, ...) and targets is a tensor of shape (batch_size, 1).

    Reports "accuracy" and "loss" metrics.
    """

    def forward(self, batch: Tuple[torch.Tensor, torch.Tensor], device: torch.device, backprop: bool) -> List[Metric]:
        if backprop:
            self.optimizer.zero_grad()

        inputs, targets = batch
        batch_size = inputs.shape[0]
        inputs, targets = inputs.to(device), targets.to(device)

        outputs = self.model(inputs)
        loss = self.criterion(outputs, targets)

        if backprop:
            loss.backward()
            self.optimizer.step()

        loss = loss.item()
        accuracy = (outputs.argmax(dim=1) == targets).sum().item() / targets.shape[0]

        # Report metrics back to the trainer.
        return [
            Metric("loss", loss, weight=batch_size),
            Metric("accuracy", accuracy, weight=batch_size),
        ]


class SupervisedBinaryClassificationTask(SupervisedLearningTask):
    """
    Expects batch to be a tuple of (inputs, targets) where inputs is a tensor of
    shape (batch_size, num_features) and targets is a tensor of shape
    (batch_size,).

    Reports "accuracy" and "loss" metrics.
    """

    def forward(self, batch: Tuple[torch.Tensor, torch.Tensor], device: torch.device, backprop: bool) -> List[Metric]:
        if backprop:
            self.optimizer.zero_grad()

        inputs, targets = batch
        batch_size = inputs.shape[0]
        inputs, targets = inputs.to(device), targets.to(device)

        outputs = self.model(inputs)
        loss = self.criterion(outputs, targets.float())

        if backprop:
            loss.backward()
            self.optimizer.step()

        loss = loss.item()
        accuracy = ((outputs > 0.5) == targets).sum().item() / targets.shape[0]

        # Report metrics back to the trainer.
        return [
            Metric("loss", loss, weight=batch_size),
            Metric("accuracy", accuracy, weight=batch_size),
        ]


class ReccurentLanguageModeling(SupervisedLearningTask):
    """
    Expects a single tensor of padded sequences of shape `(batch_size,
    max_sequence_len)`. Each sequence consist of a torch long tensor of token
    IDs.

    Each sequence is used to make multiple predictions, using teacher forcing.

    The model is expected to return a tuple of (outputs, hidden) where outputs
    is a tensor of shape (batch_size, max_sequence_len, num_classes) and hidden
    is a tensor of shape (num_layers, batch_size, hidden_size).

    Reports "perplexity" and "loss" metrics.
    """

    def forward(self, batch: torch.Tensor, device: torch.device, backprop: bool) -> List[Metric]:
        if backprop:
            self.optimizer.zero_grad()

        hidden = None

        batch = batch.to(device)
        batch_size, max_sequence_len = batch.shape

        # Shift the batch by 1 position to create the target batch
        input_batch = batch[:, :-1]
        target_batch = batch[:, 1:]

        outputs, hidden = self.model(input_batch, hidden)

        # Calculate the loss for the entire batch
        loss = self.criterion(outputs.view(-1, outputs.shape[-1]), target_batch.view(-1))

        # Backward pass
        if backprop:
            loss.backward()
            self.optimizer.step()

        return [
            Metric("loss", loss.item(), weight=batch_size * max_sequence_len),
            Metric("perplexity", math.exp(min(loss.item(), 100)), weight=batch_size * max_sequence_len),
        ]


class LanguageModelingTask(SupervisedLearningTask):
    """
    Expects a single tensor of padded sequences of shape `(batch_size,
    max_sequence_len)`. Each sequence consist of a torch long tensor of token
    IDs.

    For all sequences in the batch, the model is trained to predict the next
    token in the sequence.

    Each sequence is used to make multiple predictions, using teacher forcing.

    Reports "perplexity" and "loss" metrics.
    """

    def forward(self, batch: torch.Tensor, device: torch.device, backprop: bool):
        pass
