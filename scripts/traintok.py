# This script trains a tokenizer on the dataset.

import argparse
import glob
import logging
import boto3
import os
import sys
import pandas

# Add the .. directory to the Python path
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), ".."))

from tokenizer.tokenizer import GoScannerTokeniezr, HFTokenizer, GoAstTokenizer

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--bucket", type=str, required=True, help="Name of the S3 bucket to download the dataset files from.")
    parser.add_argument("--region", type=str, default="ap-east-1", help="Name of the AWS region to use.")
    parser.add_argument("--cache-dir", type=str, default="data", help="Local mirror of the S3 bucket. Files are not downloaded if they already exist in the cache.")
    parser.add_argument("--source-prefix", type=str, help="Feed all files in that S3 directory to the job.")
    parser.add_argument("--out", type=str, help="Output filepath.")
    parser.add_argument("--tokenizer", type=str, default="go-scanner", help="Which tokenizer to use. One of 'hf', 'go-scanner', 'go-ast'.")
    parser.add_argument("--vocab-size", type=int, default=2**15, help="Vocabulary size to use.")
    args = parser.parse_args()

    s3 = boto3.resource('s3', region_name=args.region)

    # Check that {bucket}/{source-prefix} exists
    bucket = s3.Bucket(args.bucket)
    if not bucket:
        raise Exception(f"Bucket '{args.bucket}' does not exist")
    if not bucket.objects.filter(Prefix=args.source_prefix):
        raise Exception(f"Prefix '{args.source_prefix}' does not exist in bucket '{args.bucket}'")

    # Create the cache dir if it doesn't exist
    os.makedirs(os.path.join(args.cache_dir, args.source_prefix), exist_ok=True)

    # Create the directories along the output filepath if they don't exist
    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    # Ensure {cache-dir}/{source-prefix} is either empty or matches the contents
    # of {bucket}/{source-prefix}
    logging.info(f"Syncing s3://{args.bucket}/{args.source_prefix} to {args.cache_dir}/{args.source_prefix}")
    for obj in bucket.objects.filter(Prefix=args.source_prefix):
        local_path = os.path.join(args.cache_dir, obj.key)
        if not os.path.exists(local_path):
            logging.info(f"Downloading missing file s3://{args.bucket}/{obj.key} to {local_path}")
            bucket.download_file(obj.key, local_path)

    # Instantiate the tokenizer
    if args.tokenizer == "go-scanner":
        tokenizer = GoScannerTokeniezr(vocab_size=args.vocab_size)
    elif args.tokenizer == "go-ast":
        tokenizer = GoAstTokenizer(vocab_size=args.vocab_size)
    elif args.tokenizer == "hf-tokenizer":
        tokenizer = HFTokenizer(vocab_size=args.vocab_size)
    else:
        raise Exception(f"Unknown tokenizer '{args.tokenizer}'")

    # Train the tokenizer
    files = glob.glob(os.path.join(args.cache_dir, args.source_prefix, "*.parquet"))
    files = sorted(files)
    for file in files:
        dataset_shard = pandas.read_parquet(file)
        tokenizer.train_from_iterator(dataset_shard["content"].tolist())
        logging.info(f"Training on '{file}'")

    tokenizer.save(args.out)
