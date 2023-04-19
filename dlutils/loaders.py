from concurrent.futures import ThreadPoolExecutor
from queue import SimpleQueue
import sys
from threading import Event, Semaphore, Thread
from typing import Callable, Iterable, Tuple

import torch


class StreamDataLoader:
    """
    This data loader iteratively prepares and preprocess the next `buffer_size`
    items of the `iter` in a separate thread.

    It then yields batches of size `batch_size` until the end of the `iter` is
    reached.

    Sequences are padded to `max_sequence_len` with `pad_token_id` and small
    sequences are packed into a single sequence by adding a `sep_token_id`
    between them.
    """

    def __init__(
        self,
        iter: Iterable,
        batch_size: int,
        tokenizer_fn: Callable,
        sep_token_id: int,
        pad_token_id: int,
        max_sequence_len: int,
        buffer_size: int = 10,
    ):
        self.iter = iter
        self.batch_size = batch_size
        self.tokenizer_fn = tokenizer_fn
        self.sep_token_id = sep_token_id
        self.pad_token_id = pad_token_id
        self.max_sequence_len = max_sequence_len
        self.buffer_size = buffer_size
        self.data_queue = SimpleQueue()
        self.stop_event = Event()
        self.buffer_semaphore = Semaphore(buffer_size)

        self.executor = None

        # Set custom exception handler
        sys.excepthook = self._exception_handler

    # Custom exception handler
    def _exception_handler(self, exc_type, exc_value, traceback):
        self.cleanup()
        sys.__excepthook__(exc_type, exc_value, traceback)

    def cleanup(self):
        self.stop_event.set()
        if self.executor is not None and self.executor.is_alive():
            self.executor.join()

    def _process_sequence(self, sequence: str) -> torch.Tensor:
        return torch.tensor(self.tokenizer_fn(sequence))

    def _data_loading_loop(self):
        batch_tokens = []
        concat_sequence = torch.tensor([], dtype=torch.long)

        for i, item in enumerate(self.iter):
            # print(f"Loading item {i}")
            if self.stop_event.is_set():
                break

            tokens = self._process_sequence(item)

            # If the current sequence is too long, we continuously split it.
            while tokens.numel() != 0:
                # print(f"Processing item {i} with {tokens.numel()} tokens, ({concat_sequence.numel()} in concat_sequence))")
                if concat_sequence.numel() == 0:
                    split = min(self.max_sequence_len, tokens.numel())
                    concat_sequence = tokens[:split]
                    tokens = tokens[split:]
                elif concat_sequence.numel() + 1 == self.max_sequence_len:
                    concat_sequence = torch.cat([concat_sequence, torch.tensor([self.pad_token_id], dtype=torch.long)]) 
                else:
                    split = min(self.max_sequence_len - concat_sequence.numel() - 1, tokens.numel())
                    concat_sequence = torch.cat([concat_sequence, torch.tensor([self.sep_token_id], dtype=torch.long), tokens[:split]])
                    tokens = tokens[split:]

                # If the concat_sequence is full, we flush it
                if concat_sequence.numel() == self.max_sequence_len:
                    batch_tokens.append(concat_sequence)
                    concat_sequence = torch.tensor([], dtype=torch.long)

                    if len(batch_tokens) == self.batch_size:
                        self.buffer_semaphore.acquire()
                        self.data_queue.put(torch.stack(batch_tokens))
                        batch_tokens = []

        if concat_sequence.numel() != 0:
            # Pad the sequence
            if concat_sequence.numel() < self.max_sequence_len:
                concat_sequence = torch.cat([concat_sequence, torch.full((self.max_sequence_len - concat_sequence.numel(),), self.pad_token_id, dtype=torch.long)])
            batch_tokens.append(concat_sequence)

        if batch_tokens:
            self.buffer_semaphore.acquire()
            self.data_queue.put(torch.stack(batch_tokens))

        # Add a None to the queue to signal the end of the data loading
        self.data_queue.put(None)

    def __iter__(self):
        # Start the data loading thread
        self.executor = Thread(target=self._data_loading_loop)
        self.executor.start()
        while True:
            batch = self.data_queue.get()
            if batch is None:
                break
            self.buffer_semaphore.release()
            yield batch
        self.cleanup()

    def __del__(self):
        # Set the stop event and wait for the data loading thread to finish
        self.cleanup()
