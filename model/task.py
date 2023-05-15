import math
from typing import List, Optional

import torch
from torch.nn import Module
from torch.nn.utils.clip_grad import clip_grad_norm_ as torch_clip_grad_norm
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler

import flame

from .debug import TrainingSampler
from .model import GopilotModel


class GopilotTask(flame.SimpleTask):
    """
    Expects a single tensor of padded sequences of shape `(batch_size,
    max_sequence_len+1)`. Each sequence consist of a torch long tensor of token
    IDs.

    For all sequences in the batch, the model is trained to predict the next
    token in the sequence. Each sequence is used to make multiple predictions.

    Reports "perplexity" and "loss" metrics.
    """
    def __init__(self, model: GopilotModel, criterion: Module, optimizer: Optimizer, scheduler: Optional[LRScheduler] = None, clip_gradients: Optional[float] = None, sampler: Optional[TrainingSampler] = None):
        super().__init__(model, criterion, optimizer, scheduler)
        self.clip_gradients = clip_gradients
        self.sampler = sampler
        self.step_loss = []

    def forward(self, batch: torch.Tensor, device: torch.device, backprop: bool):
        if backprop:
            self.optimizer.zero_grad()

        batch = batch.to(device)
        batch_size, sequence_length = batch.shape[0], batch.shape[1]-1

        # Shift the batch by 1 position to create the target batch
        inputs = batch[:, :-1] # (batch_size, sequence_len)
        targets = batch[:, 1:] # (batch_size, sequence_len)

        outputs = self.model(inputs) # (batch_size, sequence_len, vocab_size)
        logits = outputs.logits # (batch_size, sequence_len, vocab_size)

        # Forward the inputs, targets and outputs to the sampler for debugging
        # purposes.
        if self.sampler is not None and backprop:
            self.sampler.feed(inputs, targets, logits)

        logits = logits.view(-1, logits.size(-1)) # (batch_size * sequence_length, vocab_size)
        targets = targets.reshape(-1) # (batch_size * sequence_length)

        # Calculate the loss for the entire batch
        loss = self.criterion(logits, targets)
        loss_value = loss.item()
        self.step_loss.append(loss_value)

        # Backward pass
        if backprop:
            loss.backward()

        return [
            flame.Metric("loss", loss_value, weight=batch_size * sequence_length),
            flame.Metric("perplexity", math.exp(min(loss_value, 100)), weight=batch_size * sequence_length),
        ]

    def step(self, device: torch.device) -> List[flame.Metric]:
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
            flame.Metric("loss", step_loss_value, weight=num_losses),
            flame.Metric("perplexity", math.exp(min(step_loss_value, 100)), weight=num_losses),
            flame.Metric("lr", self.optimizer.param_groups[0]["lr"]),
        ]
