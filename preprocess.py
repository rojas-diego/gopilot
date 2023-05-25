# This script launches the jobs defined in `preprocessing`.
import argparse
import logging
import threading
import multiprocessing
multiprocessing.set_start_method('fork') # only on unix

from dataset import *

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dest-prefix', type=str, required=True, help='Export the processed files to that S3 directory.')
    parser.add_argument('--job', type=str, required=True, help='Name of the preprocessing job to run.')
    # Optional
    parser.add_argument('--bucket', type=str, default="gopilot", help='Name of the S3 bucket to download the dataset files from.')
    parser.add_argument('--source-prefix', type=str, help='Feed all files in that S3 directory to the job.')
    parser.add_argument('--cache-dir', type=str, default=".cache", help='Local mirror of the S3 bucket. Files are not downloaded if they already exist in the cache.')
    args = parser.parse_args()

    # Set up logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')

    job_name = args.job
    delattr(args, "job")
    pool = multiprocessing.Pool()

    if job_name == "tokenize-with-huggingface":
        job = TokenizeWithHuggingFaceJob(**vars(args))
    elif job_name == "tokenize-with-gopilot":
        job = TokenizeWithGopilotJob(**vars(args))
    elif job_name == "train-huggingface-tokenizer":
        job = TrainHuggingFaceTokenizerJob(**vars(args))
    elif job_name == "train-gopilot-tokenizer":
        job = TrainGopilotTokenizerJob(**vars(args))
    elif job_name == "upload-the-stack":
        job = UploadTheStackJob(**vars(args))
    else:
        raise ValueError(f"Unknown job {args.job}")
    job.run()
