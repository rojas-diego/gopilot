from typing import List, Optional

import torch
from torch.cuda.amp import GradScaler, autocast
from torch.nn import CrossEntropyLoss
from torch.nn.utils.clip_grad import clip_grad_norm_ as torch_clip_grad_norm
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from transformers.modeling_outputs import CausalLMOutputWithCrossAttentions

import flame

from .model import GopilotModel


class GopilotTask(flame.SimpleTask):
    def __init__(self, model: GopilotModel, optimizer: Optimizer, pad_token_id: int, batch_size: int, scheduler: Optional[LRScheduler] = None, 
                 clip_gradients: Optional[float] = None, precision: torch.dtype = torch.float32):
        super().__init__(model, CrossEntropyLoss(reduction="none"), optimizer, scheduler)
        self.pad_token_id = pad_token_id
        self.batch_size = batch_size
        self.clip_gradients = clip_gradients
        self.step_loss = []
        self.precision = precision
        self.scaler = GradScaler() if precision == torch.float16 else None
        self.total_tokens_ingested = 0
        self.estimation_interval = 10

    def forward(self, batch: torch.Tensor, device: torch.device, backprop: bool):
        if backprop:
            self.optimizer.zero_grad(set_to_none=True)

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
            outputs.clear()

        loss_value = loss.item()
        self.step_loss.append(loss_value)

        # Backward pass
        if backprop:
            if self.scaler:
                self.scaler.scale(loss).backward() # type: ignore
            else:
                loss.backward()

        self.total_tokens_ingested += batch_size * sequence_length

        del loss, masked_loss, total_loss, num_active_elements, inputs, targets, attention_mask, batch, logits

        return [
            flame.Metric("loss", loss_value, weight=batch_size * sequence_length, step=self.total_tokens_ingested),
            flame.Metric("total_tokens_ingested", self.total_tokens_ingested)
        ]

    def step(self, batch: torch.Tensor, device: torch.device, epoch_idx: int, batch_idx: int, step_idx: int) -> List[flame.Metric]:
        if self.clip_gradients is not None:
            torch_clip_grad_norm(self.model.parameters(), self.clip_gradients)

        if self.scaler:
            self.scaler.step(self.optimizer, bs=self.batch_size)
            self.scaler.update()
        else:
            self.optimizer.step(bs=self.batch_size) # type: ignore

        if self.scheduler is not None:
            self.scheduler.step()

        self.optimizer.zero_grad(set_to_none=True)

        num_losses = len(self.step_loss)
        step_loss_value = sum(self.step_loss) / num_losses
        self.step_loss = []

        # Update hessian estimate
        if (step_idx+1) % self.estimation_interval == 0:
            self._update_hessian(batch)

        return [
            flame.Metric("loss", step_loss_value),
            flame.Metric("lr", self.optimizer.param_groups[0]["lr"]),
        ]

    def _update_hessian(self, batch: torch.Tensor):
        inputs = batch[:, :-1]
        outputs: CausalLMOutputWithCrossAttentions = self.model(inputs)
        logits = outputs.logits
        samp_dist = torch.distributions.Categorical(logits=logits)
        y_sample = samp_dist.sample()
        loss_sampled = torch.nn.functional.cross_entropy(logits.view(-1, logits.size(-1)), y_sample.reshape(-1))
        loss_sampled.backward()
        self.optimizer.update_hessian() # type: ignore
        self.optimizer.zero_grad(set_to_none=True)
