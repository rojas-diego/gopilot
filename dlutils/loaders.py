from concurrent.futures import ThreadPoolExecutor
from queue import SimpleQueue
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

        # Start the data loading thread
        self.executor = ThreadPoolExecutor(max_workers=1)
        self.data_loading_future = self.executor.submit(self._data_loading_loop)

    def _process_sequence(self, sequence: str) -> Tuple[torch.Tensor, int]:
        tokens = torch.tensor(self.tokenizer_fn(sequence))
        num_tokens = tokens.shape[0]
        return tokens, num_tokens

    def _data_loading_loop(self):
        batch_tokens = []
        concat_sequence = torch.tensor([], dtype=torch.long)

        for item in self.iter:
            tokens, num_tokens = self._process_sequence(item)

            if num_tokens == 0:
                continue

            # If current buffer sequence is too long, we flush it
            if concat_sequence.numel() + num_tokens + 1 > self.max_sequence_len:
                # Pad the sequence
                if concat_sequence.numel() < self.max_sequence_len:
                    concat_sequence = torch.cat([concat_sequence, torch.full(self.max_sequence_len - concat_sequence.numel(), self.pad_token_id, dtype=torch.long)])

                batch_tokens.append(concat_sequence)

                if len(batch_tokens) == self.batch_size:
                    self.data_queue.put(torch.stack(batch_tokens))
                    batch_tokens = []

                concat_sequence = tokens
            else:
                sep_token = torch.tensor([self.sep_token_id], dtype=torch.long)
                concat_sequence = torch.cat([concat_sequence, sep_token, tokens]) if concat_sequence.numel() > 0 else tokens

        if batch_tokens:
            self.data_queue.put(torch.stack(batch_tokens))

        # Signal the end of the data loading
        self.data_queue.put(None)

    def __iter__(self):
        while True:
            batch = self.data_queue.get()
            if batch is None:
                break
            yield batch
