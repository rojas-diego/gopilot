import atexit
import logging
import os
import queue
import pandas
import concurrent.futures
import boto3


class StreamParquetLoader:
    def __init__(self, bucket: str, region: str, cache_dir: str, prefix: str, batch_size: int, sequence_len: int, stride: int, max_queued_files: int = 4, max_queued_batches: int = 1000):
        self._bucket_name = bucket
        self._region = region
        self._cache_dir = cache_dir
        self._prefix = prefix
        self._batch_size = batch_size
        self._sequence_len = sequence_len
        self._stride = stride
        self._client = boto3.client('s3', region_name=region)
        self._bucket = self._client.Bucket(bucket)
        # Check that {bucket}/{prefix} exists
        if not self._bucket:
            raise Exception(f"Bucket '{bucket}' does not exist")
        if not self._bucket.objects.filter(Prefix=self._prefix):
            raise Exception(f"Prefix '{self._prefix}' does not exist in bucket '{bucket}'")
        # Create the cache dir if it doesn't exist
        os.makedirs(os.path.join(cache_dir, prefix), exist_ok=True)
        # Queue to hold file paths downloaded by the background thread
        self._file_queue = queue.Queue(maxsize=max_queued_files)
        # Queue to hold batches constructed by the background thread
        self._batch_queue = queue.Queue(maxsize=max_queued_batches)
        # Start the background threads
        self._executor = None

    def _start(self):
        logging.info("Starting stream loader threads")
        self._executor = concurrent.futures.ThreadPoolExecutor(max_workers=2)
        self._executor.submit(self._download_files)
        self._executor.submit(self._construct_batches)
        atexit.register(self.__del__)

    def _download_files(self):
        for obj in self._bucket.objects.filter(Prefix=self._prefix):
            if obj.key.endswith(".parquet"):
                # Construct the local path to the file
                local_path = os.path.join(self._cache_dir, obj.key)
                # Check if the file already exists in the cache
                if os.path.exists(local_path):
                    logging.info(f"File '{obj.key}' already exists in cache")
                    self._file_queue.put(local_path)
                logging.info(f"Downloading file '{obj.key}'")
                self._bucket.download_file(obj.key, local_path)
                self._file_queue.put(local_path)
            self._file_queue.put(None)

    def _construct_batches(self):
        while True:
            # Wait until a file is downloaded
            parquet_file = self._file_queue.get()
            if parquet_file is None:
                return
            # Construct batches from the parquet file
            for batch in self._construct_batches_from_file(parquet_file):
                self._batch_queue.put(batch)

    def _construct_batches_from_file(self, parquet_file: str):
        # Load the parquet file
        df = pandas.read_parquet(parquet_file)
        # The "tokens" column contains a list of tokens.
        for tokens in df["tokens"]:
            # Construct batches using a sliding window approach with stride
            for i in range(0, len(tokens) - self._sequence_len, self._stride):
                yield tokens[i:i+self._sequence_len]

    def __iter__(self):
        # The background downloader returns paths to downloaded parquet files.
        # If the parquet file is already in the cache, it will not be downloaded
        # again.
        # The async batch channel returns batches constructed using a
        # sliding window approach with stride. Each sequence is `sequence_len`
        # tokens long.
        if self._executor is None:
            self._start()
        while True:
            batch = self._batch_queue.get()
            if batch is None:
                return
            yield batch

    def __del__(self):
        logging.info("Shutting down stream loader")
        if self._executor is not None:
            self._executor.shutdown(wait=True)
            atexit.unregister(self.__del__)
