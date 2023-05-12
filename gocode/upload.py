# Downloads the Stack Dedup v1.2 dataset Parquet files and uploads them to an
# Amazon S3 bucket for processing.

import logging
import os
import datasets
import argparse
import boto3
import tqdm
import pandas

def num_byte_to_human_readable(num_bytes: int) -> str:
    if num_bytes < 1024:
        return f"{num_bytes} B"
    elif num_bytes < 1024**2:
        return f"{num_bytes / 1024} KB"
    elif num_bytes < 1024**3:
        return f"{num_bytes / 1024**2} MB"
    elif num_bytes < 1024**4:
        return f"{num_bytes / 1024**3} GB"
    else:
        return f"{num_bytes / 1024**4} TB"
    
def write_shard_to_s3(client, bucket: str, prefix: str, cache_dir: str, shard_idx: int, samples: list):
    filename = f"shard-{shard_idx:03}.parquet"
    filepath = os.path.join(cache_dir, filename)
    frame = pandas.DataFrame(samples)
    frame.to_parquet(filepath)
    logging.info(f"Uploading shard {shard_idx} to S3...")
    client.upload_file(filepath, bucket, f"{prefix}/{filename}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--bucket', type=str, required=True, help='Name of the S3 bucket to upload the dataset files to.')
    parser.add_argument('--prefix', type=str, default="datasets/the-stack-dedup-v1.2/base", help='Prefix to use for the S3 object keys.')
    parser.add_argument('--cache-dir', type=str, default="data", help='Path to the directory in which to cache the dataset files.')
    parser.add_argument('--num-samples-per-shard', type=int, default=4_000_000, help='Number of samples to include in each shard.')
    parser.add_argument('--num-bytes-per-shard', type=int, default=64_000_000, help='Number of bytes to include in each shard.')
    args = parser.parse_args()

    # Download the dataset files.
    the_stack_dedup_go = datasets.load_dataset('bigcode/the-stack-dedup', data_dir="data/go", split="train", use_auth_token=True)

    # Connect to S3.
    s3 = boto3.client('s3')

    current_shard_idx = 0
    current_shard_samples = []
    num_bytes_in_current_shard = 0
    num_samples_in_current_shard = 0
    
    progress_bar = tqdm.tqdm(the_stack_dedup_go, desc="Building shards")
    for sample in progress_bar:
        # Accumulate samples and bytes until we reach the shard size.
        num_bytes_in_current_shard += len(sample['content']) # type: ignore
        num_samples_in_current_shard += 1
        current_shard_samples.append(sample)

        # If we've reached the shard size, upload the shard and reset the
        # counters.
        if num_bytes_in_current_shard >= args.num_bytes_per_shard or num_samples_in_current_shard >= args.num_samples_per_shard:
            write_shard_to_s3(s3, args.bucket, args.prefix, args.cache_dir, current_shard_idx, current_shard_samples)
            current_shard_idx += 1
            current_shard_samples = []
            num_bytes_in_current_shard = 0
            num_samples_in_current_shard = 0

        # Update the progress bar.
        progress_bar.set_postfix_str(f"Current Shard Index: {current_shard_idx}, Size: {num_byte_to_human_readable(num_bytes_in_current_shard)}, Num Samples: {num_samples_in_current_shard}")
