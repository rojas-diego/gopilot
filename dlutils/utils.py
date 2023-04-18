from typing import List
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
