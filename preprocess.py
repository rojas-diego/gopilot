# This script launches the jobs defined in `preprocessing`.
import argparse
import logging

from dataset import *

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dest-prefix', type=str, required=True, help='Export the processed files to that S3 directory.')
    parser.add_argument('--job', type=str, required=True, help='Name of the preprocessing job to run.')
    # Optional
    parser.add_argument('--bucket', type=str, default="gopilot", help='Name of the S3 bucket to download the dataset files from.')
    parser.add_argument('--region', type=str, default="ap-east-1", help='Name of the AWS region to use.')
    parser.add_argument('--source-prefix', type=str, help='Feed all files in that S3 directory to the job.')
    parser.add_argument('--cache-dir', type=str, default=".cache", help='Local mirror of the S3 bucket. Files are not downloaded if they already exist in the cache.')
    args = parser.parse_args()

    # Set up logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')

    job_name = args.job
    delattr(args, "job")

    if job_name == "tokenize-hf":
        job = TokenizeWithHuggingFaceJob(**vars(args))
    elif job_name == "tokenize-go-scanner":
        job = TokenizeWithGoScannerJob(**vars(args))
    elif job_name == "train-hf-tokenizer":
        job = TrainHuggingFaceTokenizerJob(**vars(args))
    elif job_name == "train-go-scanner-tokenizer":
        job = TrainGoScannerTokenizerJob(**vars(args))
    elif job_name == "upload":
        job = UploadTheStackJob(**vars(args))
    else:
        raise ValueError(f"Unknown job {args.job}")

    job.run()
