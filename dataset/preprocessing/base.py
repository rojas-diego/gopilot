import json
import logging
import os
import pandas
import boto3
import shutil

class PreprocessingJob:
    def __init__(self, bucket: str, region: str, cache_dir: str, source_prefix: str | None, dest_prefix: str):
        self._bucket_name = bucket
        self._region = region
        self._cache_dir = cache_dir
        self._source_prefix = source_prefix
        self._dest_prefix = dest_prefix
        self._bucket = boto3.resource('s3', region_name=region).Bucket(bucket)
        # Check that {bucket}/{source-prefix} exists
        if not self._bucket:
            raise Exception(f"Bucket '{bucket}' does not exist")
        if not self._bucket.objects.filter(Prefix=self._source_prefix):
            raise Exception(f"Prefix '{self._source_prefix}' does not exist in bucket '{bucket}'")
        # Create the cache dir if it doesn't exist
        if source_prefix:
            os.makedirs(os.path.join(cache_dir, source_prefix), exist_ok=True)
        os.makedirs(os.path.join(cache_dir, dest_prefix), exist_ok=True)

    def files(self):
        """Returns an iterator that loops over the files in the given prefix.
        If the files are not locally cached, they will be downloaded."""
        assert self._source_prefix is not None
        for obj in self._bucket.objects.filter(Prefix=self._source_prefix):
            local_path = os.path.join(self._cache_dir, obj.key)
            if not os.path.exists(local_path):
                logging.info(f"Downloading missing file s3://{self._bucket_name}/{obj.key} to {local_path}")
                self._bucket.download_file(obj.key, local_path)
            else:
                logging.info(f"Using cached file {local_path}")
            yield local_path

    def file(self, filename: str):
        """Downloads the given file from S3 to the local cache and returns the
        local path."""
        assert self._source_prefix is not None
        local_path = os.path.join(self._cache_dir, self._source_prefix, filename)
        if not os.path.exists(local_path):
            logging.info(f"Downloading missing file s3://{self._bucket_name}/{self._source_prefix}/{filename} to {local_path}")
            self._bucket.download_file(os.path.join(self._source_prefix, filename), local_path)
        return local_path

    def save_parquet(self, df: pandas.DataFrame, filename: str):
        """Writes the parquet file to the local cache and uploads it to S3."""
        df.to_parquet(os.path.join(self._cache_dir, self._dest_prefix, filename))
        self._bucket.upload_file(os.path.join(self._cache_dir, self._dest_prefix, filename), os.path.join(self._dest_prefix, filename))
        logging.info(f"Uploaded file {filename} to s3://{self._bucket_name}/{self._dest_prefix}/{filename}")

    def save_json(self, data: dict, filename: str):
        """Writes the json file to the local cache and uploads it to S3."""
        with open(os.path.join(self._cache_dir, self._dest_prefix, filename), "w") as f:
            f.write(json.dumps(data))
        self._bucket.upload_file(os.path.join(self._cache_dir, self._dest_prefix, filename), os.path.join(self._dest_prefix, filename))
        logging.info(f"Uploaded file {filename} to s3://{self._bucket_name}/{self._dest_prefix}/{filename}")

    def save(self, path: str, filename: str):
        """Copies the file to the local cache and uploads it to S3."""
        shutil.copy(path, os.path.join(self._cache_dir, self._dest_prefix, filename))
        self._bucket.upload_file(path, os.path.join(self._dest_prefix, filename))
        logging.info(f"Uploaded file {filename} to s3://{self._bucket_name}/{self._dest_prefix}/{filename}")
