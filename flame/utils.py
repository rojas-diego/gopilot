import logging
import time
from typing import Any, List
import torch
from torch.nn import Module
import torch.nn as nn


class Metric:
    def __init__(self, name: str, value: float, weight: int = 1):
        self.name = name
        self.value = value
        self.weight = weight

    def __repr__(self):
        return f"{self.name}: {self.value:.4f}"


class MetricsStore:
    def __init__(self):
        self.metrics_total_sum = {}
        self.metrics_total_count = {}

    def accumulate(self, metrics: List[Metric]):
        for metric in metrics:
            if metric.name not in self.metrics_total_sum:
                self.metrics_total_sum[metric.name] = 0
                self.metrics_total_count[metric.name] = 0
            self.metrics_total_sum[metric.name] += metric.value * metric.weight
            self.metrics_total_count[metric.name] += metric.weight

    def mean(self):
        return [
            Metric(
                name,
                self.metrics_total_sum[name] / self.metrics_total_count[name],
                self.metrics_total_count[name]
            ) for name in self.metrics_total_sum
        ]


def xavier_initialization(model: Module):
    for module in model.modules():
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.BatchNorm2d):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)


def kaiming_initialization(model: Module):
    for module in model.modules():
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            nn.init.kaiming_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.BatchNorm2d):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)


def best_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():  # type: ignore
        return torch.device("mps")
    else:
        return torch.device("cpu")


def log_model_summary(model: Module):
    def human_format(num):
        if abs(num) < 1000:
            return '{:.0f}'.format(num)
        elif abs(num) < 1e6:
            return '{:.1f}k'.format(num/1000)
        elif abs(num) < 1e9:
            return '{:.1f}M'.format(num/1e6)
        else:
            return '{:.1f}B'.format(num/1e9)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logging.info(f"Model: {model.__class__.__name__}")
    logging.info(f"Total parameters: {human_format(total_params)}")
    logging.info(f"Trainable parameters: {human_format(trainable_params)}")
    logging.info(f"Non-trainable parameters: {human_format(total_params - trainable_params)}")
    logging.info(f"Total layers: {len(list(model.modules()))}")


class LinearLRScheduleWithTimeBudget:
    def __init__(self, warmup_steps: int, training_budget_secs: int, min_factor: float):
        self.warmup_steps = warmup_steps
        self.training_budget_secs = training_budget_secs
        self.min_factor = min_factor
        self.start_time = time.time()

    def __call__(self, step: int) -> float:
        if step < self.warmup_steps:
            return step / self.warmup_steps
        if step == self.warmup_steps:
            self.start_time = time.time()
        step = step - self.warmup_steps
        elapsed_time = time.time() - self.start_time
        scale_factor = 1 - (elapsed_time / self.training_budget_secs)
        return max(self.min_factor, scale_factor)
