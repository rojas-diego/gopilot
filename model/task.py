import math
from typing import List, Optional

import torch
from torch.cuda.amp import GradScaler, autocast
from torch.nn import Module
from torch.nn.utils.clip_grad import clip_grad_norm_ as torch_clip_grad_norm
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler

import flame

from .debug import TrainingSampler
from .model import GopilotModel


class GopilotTask(flame.SimpleTask):
    def __init__(self, model: GopilotModel, criterion: Module, optimizer: Optimizer, scheduler: Optional[LRScheduler] = None, 
                 clip_gradients: Optional[float] = None, sampler: Optional[TrainingSampler] = None, 
                 precision: torch.dtype = torch.float32):
        super().__init__(model, criterion, optimizer, scheduler)
        self.clip_gradients = clip_gradients
        self.sampler = sampler
        self.step_loss = []
        self.precision = precision
        self.scaler = GradScaler() if precision == torch.float16 else None

    def forward(self, batch: torch.Tensor, device: torch.device, backprop: bool):
        if backprop:
            self.optimizer.zero_grad()

        batch = batch.to(device)
        batch_size, sequence_length = batch.shape[0], batch.shape[1]-1

        # Shift the batch by 1 position to create the target batch
        inputs = batch[:, :-1] 
        targets = batch[:, 1:] 

        with autocast(self.precision == torch.float16):
            outputs = self.model(inputs)
            logits = outputs.logits 
            logits = logits.view(-1, logits.size(-1)) 
            targets = targets.reshape(-1) 

            # Calculate the loss for the entire batch
            loss: torch.Tensor = self.criterion(logits, targets)
        
        loss_value = loss.item()
        self.step_loss.append(loss_value)

        # Backward pass
        if backprop:
            if self.scaler:
                self.scaler.scale(loss).backward() # type: ignore
            else:
                loss.backward()

        return [
            flame.Metric("loss", loss_value, weight=batch_size * sequence_length),
            flame.Metric("perplexity", math.exp(min(loss_value, 100)), weight=batch_size * sequence_length),
        ]

    def step(self, device: torch.device) -> List[flame.Metric]:
        if self.clip_gradients is not None:
            torch_clip_grad_norm(self.model.parameters(), self.clip_gradients)

        if self.scaler:
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
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
