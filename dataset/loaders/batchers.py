from abc import ABC, abstractmethod
from ast import List
import random
from typing import Iterable, Iterator, List, Tuple

import torch


class Batcher(ABC):
    """Generates batches of data from samples."""
    @abstractmethod
    def batches(self, samples: Iterable) -> Iterable:
        pass


class VariableLengthStridedWindowBatcher(Batcher):
    def __init__(self, batch_size: int, window_size: int, pad_token_id: int, eos_token_id: int, stride_range: Tuple[int, int] = (1, 10)):
        self.batch_size = batch_size
        self.window_size = window_size
        self.pad_token_id = pad_token_id
        self.eos_token_id = eos_token_id
        self.stride_range = stride_range

    def batches(self, dataset_samples: Iterator[List[int]]) -> Iterable:
        """Yields batches of data from a DataFrame."""
        tokens_chunks = []
        batche_samples = []
        for tokens in dataset_samples:
            # Accumulate at least 100 times batch_size lists of tokens
            tokens_chunks.append(tokens)
            if len(tokens_chunks) < 100*self.batch_size:
                continue
            # For each chunk of tokens, form batches of sequences
            for tokens in tokens_chunks:
                # Left pad the sequence with window_size-1 padding tokens
                tokens = [self.pad_token_id]*(self.window_size-1) + tokens
                # Right pad with end of sequence token
                tokens = tokens + [self.eos_token_id]
                for i in range(0, len(tokens) - self.window_size, self.stride_range[0] + random.randint(1, self.stride_range[1]) - 1):
                    sequence = torch.tensor(tokens[i:i+self.window_size], dtype=torch.long)
                    batche_samples.append(sequence)
            # Using this batches list, yield 100 batches of sequences
            random.shuffle(batche_samples)
            for i in range(0, len(batche_samples), self.batch_size):
                yield torch.stack(batche_samples[i:i+self.batch_size])
            # Reset the tokens_chunks and batches lists
            tokens_chunks = []
            batche_samples = []
        # TODO: Yield the remaining batches
