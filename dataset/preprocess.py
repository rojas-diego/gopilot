# This script launches the jobs defined in `preprocessing`.

import argparse
import logging

from .preprocessing.tokenize import (GoAstTokenizeJob, GoScannerTokenizeJob, HFTokenizeJob)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--bucket', type=str, required=True, help='Name of the S3 bucket to download the dataset files from.')
    parser.add_argument('--region', type=str, default="ap-east-1", help='Name of the AWS region to use.')
    parser.add_argument('--source-prefix', type=str, help='Feed all files in that S3 directory to the job.')
    parser.add_argument('--dest-prefix', type=str, required=True, help='Export the processed files to that S3 directory.')
    parser.add_argument('--cache-dir', type=str, default=".cache", help='Local mirror of the S3 bucket. Files are not downloaded if they already exist in the cache.')
    parser.add_argument('--job', type=str, required=True, help='Name of the preprocessing job to run.')
    args = parser.parse_args()

    # Set up logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')

    if args.job == "tokenize-hf":
        job = HFTokenizeJob(**vars(args))
    elif args.job == "tokenize-go-ast":
        job = GoAstTokenizeJob(**vars(args))
    elif args.job == "tokenize-go-scanner":
        job = GoScannerTokenizeJob(**vars(args))
    else:
        raise ValueError(f"Unknown job {args.job}")
