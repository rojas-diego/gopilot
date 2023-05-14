from abc import ABC, abstractmethod
from ast import List
import random
from typing import Iterable, Iterator, List

import pandas
import torch


class Batcher(ABC):
    """Given a DataFrame, forms batches of data."""
    @abstractmethod
    def batches(self, samples: Iterable) -> Iterable:
        pass


class StridedWindowBatcher(Batcher):
    def __init__(self, batch_size: int, window_size: int, stride: int):
        self.batch_size = batch_size
        self.window_size = window_size
        self.stride = stride

    def batches(self, samples: Iterator[List[int]]) -> Iterable:
        """Yields batches of data from a DataFrame."""
        batch = []
        for tokens in samples:
            for i in range(0, len(tokens) - self.window_size, self.stride):
                sequence = torch.tensor(tokens[i:i+self.window_size], dtype=torch.long)
                batch.append(sequence)
                if len(batch) == self.batch_size:
                    yield torch.stack(batch)
                    batch = []
        if len(batch) > 0:
            yield torch.stack(batch)
