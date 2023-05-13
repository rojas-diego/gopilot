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
    if num_bytes < 1000:
        return f"{num_bytes} B"
    elif num_bytes < 1000**2:
        return f"{num_bytes / 1000} KB"
    elif num_bytes < 1000**3:
        return f"{num_bytes / 1000**2} MB"
    elif num_bytes < 1000**4:
        return f"{num_bytes / 1000**3} GB"
    else:
        return f"{num_bytes / 1000**4} TB"

def write_shard_to_s3(client, bucket: str, prefix: str, cache_dir: str, shard_idx: int, samples: list):
    filename = f"shard-{shard_idx:03}.parquet"
    filepath = os.path.join(os.path.join(cache_dir, prefix), filename)
    frame = pandas.DataFrame(samples)
    frame.to_parquet(filepath, compression="brotli")
    logging.info(f"Uploading shard {shard_idx} to S3...")
    client.upload_file(filepath, bucket, f"{prefix}/{filename}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--bucket', type=str, required=True, help='Name of the S3 bucket to upload the dataset files to.')
    parser.add_argument('--region', type=str, default="ap-east-1", help='Name of the AWS region to use.')
    parser.add_argument('--prefix', type=str, default="datasets/the-stack-dedup-v1.2/base", help='Prefix to use for the S3 object keys.')
    parser.add_argument('--cache-dir', type=str, default="data", help='Path to the directory in which to cache the dataset files.')
    parser.add_argument('--num-samples-per-shard', type=int, default=None, help='Number of samples to include in each shard.')
    parser.add_argument('--num-bytes-per-shard', type=int, default=None, help='Number of bytes to include in each shard.')
    args = parser.parse_args()

    if args.num_samples_per_shard is None and args.num_bytes_per_shard is None:
        raise ValueError("Must specify either --num-samples-per-shard or --num-bytes-per-shard.")
    
    # Create the cache directory if it doesn't exist.
    os.makedirs(os.path.join(args.cache_dir, args.prefix), exist_ok=True)

    # If there's data already in the cache directory, warn the user.
    if len(os.listdir(os.path.join(args.cache_dir, args.prefix))) > 0:
        raise ValueError("Cache directory is not empty. Please clear it before running this script.")

    # Download the dataset files.
    the_stack_dedup_go = datasets.load_dataset('bigcode/the-stack-dedup', data_dir="data/go", split="train", use_auth_token=True)

    # Connect to S3.
    s3 = boto3.client('s3', region_name=args.region)

    current_shard_idx = 0
    current_shard_samples = []
    num_bytes_in_current_shard = 0
    num_samples_in_current_shard = 0
    
    progress_bar = tqdm.tqdm(the_stack_dedup_go, desc="Building shards", mininterval=1)
    for sample in progress_bar:
        # Accumulate samples and bytes until we reach the shard size.
        num_bytes_in_current_shard += len(sample['content']) # type: ignore
        num_samples_in_current_shard += 1
        current_shard_samples.append(sample)

        # If we've reached the shard size, upload the shard and reset the
        # counters.
        if (args.num_bytes_per_shard is not None and num_bytes_in_current_shard >= args.num_bytes_per_shard) or (args.num_samples_per_shard is not None and num_samples_in_current_shard >= args.num_samples_per_shard):
            write_shard_to_s3(s3, args.bucket, args.prefix, args.cache_dir, current_shard_idx, current_shard_samples)
            current_shard_idx += 1
            current_shard_samples = []
            num_bytes_in_current_shard = 0
            num_samples_in_current_shard = 0

        # Update the progress bar.
        progress_bar.set_postfix_str(f"Current Shard Index: {current_shard_idx}, Size: {num_byte_to_human_readable(num_bytes_in_current_shard)}, Num Samples: {num_samples_in_current_shard}", refresh=False)

    # If there are any samples left over, upload them as a final shard.
    if len(current_shard_samples) > 0:
        write_shard_to_s3(s3, args.bucket, args.prefix, args.cache_dir, current_shard_idx, current_shard_samples)
