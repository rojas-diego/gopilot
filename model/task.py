import math
from typing import List, Optional

import torch
from torch.cuda.amp import GradScaler, autocast
from torch.nn import CrossEntropyLoss
from torch.nn.utils.clip_grad import clip_grad_norm_ as torch_clip_grad_norm
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from transformers.modeling_outputs import CausalLMOutputWithCrossAttentions

import flame

from .debug import TrainingSampler
from .model import GopilotModel


class GopilotTask(flame.SimpleTask):
    def __init__(self, model: GopilotModel, optimizer: Optimizer, pad_token_id: int, scheduler: Optional[LRScheduler] = None, 
                 clip_gradients: Optional[float] = None, sampler: Optional[TrainingSampler] = None, 
                 precision: torch.dtype = torch.float32):
        super().__init__(model, CrossEntropyLoss(reduce="none"), optimizer, scheduler)
        self.pad_token_id = pad_token_id
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

        # Create attention mask
        attention_mask = (inputs != self.pad_token_id).long()

        with autocast(self.precision == torch.float16):
            outputs: CausalLMOutputWithCrossAttentions = self.model(inputs, attention_mask=attention_mask)
            logits = outputs.logits

            # Calculate the masked loss
            loss_mask = (targets != self.pad_token_id).float()
            loss = self.criterion(logits.view(-1, logits.size(-1)), targets.reshape(-1))
            masked_loss = loss * loss_mask.view(-1)
            total_loss = torch.sum(masked_loss)
            num_active_elements = torch.sum(loss_mask)
            if num_active_elements == 0:
                num_active_elements = torch.tensor(1e-8)
            loss = total_loss / num_active_elements

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
            flame.Metric("tokens_ingested", batch_size * sequence_length)
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
