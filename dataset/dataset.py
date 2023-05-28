import logging
import os
import random

import boto3
import numpy
import torch
from torch.utils.data import IterableDataset, Dataset

from tokenizer import Tokenizer


class GopilotDataset(IterableDataset):
    def __init__(self, bucket: str, prefix: str, cache_dir: str, window_size: int, stride: int, min_concurrent_samples: int = -1):
        self.bucket_name = bucket
        self.prefix = prefix
        self.cache_dir = cache_dir
        self.window_size = window_size
        self.stride = stride
        # Keep a minimum of 1GB in memory.
        # window_size * sizeof(int64) * min_concurrent_samples >= 1GB
        self.min_concurrent_samples = min_concurrent_samples if min_concurrent_samples > 0 else 1024 * 1024 * 1024 // (window_size * 8)
        # Ensure that the cache directory exists.
        os.makedirs(os.path.join(cache_dir, prefix), exist_ok=True)

    def _iter_remote_files(self):
        self.bucket = boto3.resource("s3").Bucket(self.bucket_name) # type: ignore
        remote_files = [obj.key for obj in self.bucket.objects.filter(Prefix=self.prefix)]
        remote_files = [remote_file for remote_file in remote_files if remote_file.endswith(".npy")]
        logging.info(f"Found {len(remote_files)} files in '{self.bucket.name}/{self.prefix}'")
        random.shuffle(remote_files)

        for remote_file in remote_files:
            local_file = os.path.join(self.cache_dir, remote_file)
            if not os.path.exists(local_file):
                logging.info(f"Downloading '{remote_file}' to '{local_file}'")
                self.bucket.download_file(remote_file, local_file)
            else:
                logging.info(f"Using cached file '{local_file}'")
            yield local_file


    def __iter__(self):
        current_sample_id = 0
        samples = {}
        # Keep a maximum of min_concurrent_samples in memory and randomly sample from them.
        for local_file in self._iter_remote_files():
            while len(samples) >= self.min_concurrent_samples:
                yield samples.pop(random.choice(list(samples.keys())))
            else:
                inital_id = current_sample_id
                mapped_file: numpy.ndarray = numpy.load(local_file)
                logging.info(f"Loaded '{local_file}' with shape {mapped_file.shape}, {mapped_file.nbytes / 1024 / 1024:.2f}MB")
                for i in range(0, mapped_file.shape[0] - self.window_size, self.stride):
                    samples[current_sample_id] = torch.from_numpy(mapped_file[i:i+self.window_size].astype(numpy.int64)).to(torch.long)
                    current_sample_id += 1
                logging.info(f"Generated {current_sample_id - inital_id} samples from '{local_file}'")
                del mapped_file
        while len(samples) > 0:
            yield samples.pop(random.choice(list(samples.keys())))


class GopilotFineTuningDataset(Dataset):
    """
    A fine-tuning dataset can be built from a JSONL file containing samples of
    the following form:

    ```
    {"sample": "package main\\n\\nconst HelloWorld = 1"}
    ```
    """
    def __init__(self, filepath: str, tokenizer: Tokenizer, window_size: int, stride: int):
        self.filepath = filepath
        self.tokenizer = tokenizer
        self.window_size = window_size
        self.stride = stride
        self.pad_token_id = tokenizer.special_token_to_id("[PAD]")
        self.samples = self._build_samples()

    def _build_samples(self):
        samples = []
        with open(self.filepath, "r") as f:
            for line in f:
                sample = eval(line)
                samples.append(sample)
        # Tokenize and prepare sequences of shape (window_size,)


        return samples
    