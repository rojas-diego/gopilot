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
        many_token_sequences = []
        batch_of_batches = []
        for token_sequence in dataset_samples:
            # Accumulate at least 64 times batch_size lists of tokens
            many_token_sequences.append(token_sequence)
            if len(many_token_sequences) < 64*self.batch_size:
                continue
            # For each chunk of tokens, form batches of sequences
            for token_sequence in many_token_sequences:
                # Right pad with end of sequence token
                token_sequence = token_sequence + [self.eos_token_id]
                for i in range(0, len(token_sequence), self._random_stride()):
                    sequence = token_sequence[i:i+self.window_size]
                    sequence = sequence + [self.pad_token_id] * (self.window_size - len(sequence))
                    assert len(sequence) == self.window_size
                    sequence = torch.tensor(sequence, dtype=torch.long)
                    batch_of_batches.append(sequence)
            # Using this batches list, yield 64 batches of sequences
            random.shuffle(batch_of_batches)
            for i in range(0, len(batch_of_batches), self.batch_size):
                yield torch.stack(batch_of_batches[i:i+self.batch_size])
            # Reset the many_token_sequences and batches lists
            many_token_sequences = []
            batch_of_batches = []
        # TODO: Yield the remaining batches

    def _random_stride(self):
        return random.randint(*self.stride_range)
