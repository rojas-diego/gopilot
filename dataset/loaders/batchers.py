from abc import ABC, abstractmethod
from ast import List
import random
from typing import Iterable, Iterator, List

import torch


class Batcher(ABC):
    """Generates batches of data from samples."""
    @abstractmethod
    def batches(self, samples: Iterable) -> Iterable:
        pass


class StridedWindowBatcher(Batcher):
    def __init__(self, batch_size: int, window_size: int, stride: int = 1):
        self.batch_size = batch_size
        self.window_size = window_size
        self.stride = stride

    def batches(self, samples: Iterator[List[int]]) -> Iterable:
        """Yields batches of data from a DataFrame."""
        tokens_chunks = []
        batches = []
        for tokens in samples:
            # Accumulate at least 100 times batch_size lists of tokens
            tokens_chunks.append(tokens)
            if len(tokens_chunks) < 100*self.batch_size:
                continue
            # For each chunk of tokens, form batches of sequences
            for tokens in tokens_chunks:
                for i in range(0, len(tokens) - self.window_size, self.stride):
                    sequence = torch.tensor(tokens[i:i+self.window_size], dtype=torch.long)
                    batches.append(sequence)
            # Using this batches list, yield 100 batches of sequences
            random.shuffle(batches)
            for i in range(0, len(batches), self.batch_size):
                yield torch.stack(batches[i:i+self.batch_size])    
            # Reset the tokens_chunks and batches lists
            tokens_chunks = []
            batches = []
        # Yield the remaining batches
        random.shuffle(batches)
        for i in range(0, len(batches), self.batch_size):
            yield torch.stack(batches[i:i+self.batch_size])
