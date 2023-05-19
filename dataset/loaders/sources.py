import glob
import logging
import os
import random
from abc import ABC, abstractmethod
from typing import Callable, Iterable, Optional, Union

import boto3

import flame


class DataSource(ABC):
    """Yields filepaths that represent chunks of a dataset."""
    @abstractmethod
    def files(self) -> Iterable[str]:
        pass


class CachedS3DataSource(DataSource):
    def __init__(self, bucket: str, cache_dir: str, prefix: str, file_lambda: Callable = lambda x: x.endswith(".parquet"), shuffle: bool = True, tracker: Optional[Union[flame.NeptuneTracker, flame.NoopTracker]] = None):
        self.cache_dir = cache_dir
        self.prefix = prefix
        self.file_lambda = file_lambda
        self.shuffle = shuffle
        self.bucket = boto3.resource('s3').Bucket(bucket)
        self.tracker = tracker
        self.shards_visited = 0
        # Ensure the cache directory exists
        os.makedirs(os.path.dirname(f"{self.cache_dir}/{self.prefix}"), exist_ok=True)

    def files(self) -> Iterable[str]:
        """Iterates over all the files in a {bucket}/{prefix} directory.
        Downloads them to {cache_dir} if they don't already exist."""
        remote_files = list(self.bucket.objects.filter(Prefix=self.prefix))
        if self.shuffle:
            random.shuffle(remote_files)
        for remote_file in remote_files:
            self.shards_visited += 1
            if self.file_lambda(remote_file.key):
                local_file = f"{self.cache_dir}/{remote_file.key}"
                if not os.path.exists(local_file):
                    os.makedirs(os.path.dirname(local_file), exist_ok=True)
                    logging.info(f"Downloading s3://{self.bucket.name}/{remote_file.key} to {local_file}")
                    self.bucket.download_file(remote_file.key, f"{self.cache_dir}/{remote_file.key}")
                else:
                    logging.info(f"Skipping download s3://{self.bucket.name}/{remote_file.key}, already exists")
                if self.tracker:
                    self.tracker.track_values([flame.Metric("dataset/shards_visited", self.shards_visited)])
                    self.tracker.track_log("dataset/files_visited", f"s3://{self.bucket.name}/{remote_file.key}")
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
