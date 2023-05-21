import logging
import os
import boto3
import numpy
import torch
from torch.utils.data import IterableDataset


class DistributedGopilotDataset(IterableDataset):
    def __init__(self, bucket: str, prefix: str, cache_dir: str, window_size: int, stride: int, rank: int = 0, world_size: int = 1, device: torch.device = torch.device("cpu")):
        self.bucket_name = bucket
        self.prefix = prefix
        self.cache_dir = cache_dir
        self.window_size = window_size
        self.stride = stride
        self.rank = rank
        self.world_size = world_size
        # Ensure that the cache directory exists.
        os.makedirs(os.path.join(cache_dir, prefix), exist_ok=True)

    def __iter__(self):
        self.bucket = boto3.resource("s3").Bucket(self.bucket_name)
        remote_files = [obj.key for obj in self.bucket.objects.filter(Prefix=self.prefix)]
        remote_files = [remote_file for remote_file in remote_files if remote_file.endswith(".npy")]
        logging.info(f"Found {len(remote_files)} files in '{self.bucket.name}/{self.prefix}'")
        remote_files = remote_files[self.rank::self.world_size]
        logging.info(f"Using {len(remote_files)} files for rank {self.rank} of {self.world_size}")
        for remote_file in remote_files:
            local_file = os.path.join(self.cache_dir, remote_file)
            if not os.path.exists(local_file):
                logging.info(f"Downloading '{remote_file}' to '{local_file}'")
                self.bucket.download_file(remote_file, local_file)
            else:
                logging.info(f"Using cached chunk '{local_file}'")
            yield from self._iter_local_file(local_file)

    def _iter_local_file(self, local_file: str):
        mapped_file: numpy.ndarray = numpy.load(local_file)
        logging.info(f"Loaded '{local_file}' with shape {mapped_file.shape}, {mapped_file.nbytes / 1024 / 1024:.2f}MB")
        for i in range(0, mapped_file.shape[0] - self.window_size, self.stride):
            yield torch.from_numpy(mapped_file[i:i+self.window_size].astype(numpy.int32)).to(torch.long)
