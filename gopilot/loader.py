# Implementation of a data loader for the GoPilot dataset.

import logging
import pyarrow.parquet as parquet
from tokenizers import Tokenizer
import torch
from torch.utils.data import IterableDataset


class GopilotDataset(IterableDataset):
    """
    Progressively loads the Parquet files in the provided list into memory and
    yields the resulting data as a stream of samples.

    Yields Torch tensors of shape (sequence_length+1). The input sequence is the
    [0:sequence_length] slice, and the target sequence is the
    [1:sequence_length+1] slice.
    """

    def __init__(self, files: list[str], tokenizer: Tokenizer, sequence_length: int):
        logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
        self.files = files
        self.tokenizer = tokenizer
        self.sequence_length = sequence_length
        logging.info(f'Initialized dataset with {len(files)} files')

    def __iter__(self):
        for tokens in self._consume():
            # Generate samples of length sequence_length+1 using a sliding window
            for i in range(len(tokens) - self.sequence_length):
                yield torch.tensor(tokens[i:i + self.sequence_length + 1])

    def _consume(self):
        for file in self.files:
            chunk = parquet.read_table(file)
            logging.info(f'Processing chunk \'{file}\' ({chunk.num_rows} samples)')
            chunk = chunk.to_pandas()
            for index, row in chunk.iterrows():
                yield self.tokenizer.encode(row['content']).ids
