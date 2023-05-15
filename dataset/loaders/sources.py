import glob
import logging
import os
from abc import ABC, abstractmethod
import random
from typing import Callable, Iterable

import boto3


class DataSource(ABC):
    """Yields filepaths that represent chunks of a dataset."""
    @abstractmethod
    def files(self) -> Iterable[str]:
        pass

class CachedS3DataSource(DataSource):
    def __init__(self, bucket: str, cache_dir: str, prefix: str, file_lambda: Callable = lambda x: x.endswith(".parquet"), shuffle: bool = True):
        self.cache_dir = cache_dir
        self.prefix = prefix
        self.file_lambda = file_lambda
        self.shuffle = shuffle
        self.bucket = boto3.resource('s3').Bucket(bucket)
        # Ensure the cache directory exists
        os.makedirs(os.path.dirname(f"{self.cache_dir}/{self.prefix}"), exist_ok=True)

    def files(self) -> Iterable[str]:
        """Iterates over all the files in a {bucket}/{prefix} directory.
        Downloads them to {cache_dir} if they don't already exist."""
        remote_files = list(self.bucket.objects.filter(Prefix=self.prefix))
        if self.shuffle:
            random.shuffle(remote_files)
        for remote_file in remote_files:
            if self.file_lambda(remote_file.key):
                local_file = f"{self.cache_dir}/{remote_file.key}"
                if not os.path.exists(local_file):
                    logging.info(f"Downloading s3://{self.bucket.name}/{remote_file.key} to {local_file}")
                    self.bucket.download_file(remote_file.key, f"{self.cache_dir}/{remote_file.key}")
                else:
                    logging.info(f"Skipping download s3://{self.bucket.name}/{remote_file.key}, already exists")
                yield local_file

class LocalGlobDataSource(DataSource):
    def __init__(self, glob_pattern: str, shuffle: bool = True):
        self.glob_pattern = glob_pattern
        self.shuffle = shuffle

    def files(self) -> Iterable[str]:
        """Iterates over all the files in the glob pattern."""
        local_files = glob.glob(self.glob_pattern)
        if self.shuffle:
            random.shuffle(local_files)
        for local_file in local_files:
            logging.info(f"Found local file {local_file}")
            yield local_file
