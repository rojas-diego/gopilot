# This script serves as an interface to launch the preprocessing jobs defined in
# `gocode`.
# Specify a source dataset from S3, and a destination dataset to upload to S3
# as well as the preprocessing script to use.

import argparse
import glob
import importlib
import json
import logging
import boto3
import os
import sys

# Add the .. directory to the Python path
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), ".."))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--bucket', type=str, required=True, help='Name of the S3 bucket to download the dataset files from.')
    parser.add_argument('--region', type=str, default="ap-east-1", help='Name of the AWS region to use.')
    parser.add_argument('--source-prefix', type=str, help='Feed all files in that S3 directory to the job.')
    parser.add_argument('--dest-prefix', type=str, help='Export the processed files to that S3 directory.')
    parser.add_argument('--cache-dir', type=str, default="data", help='Local mirror of the S3 bucket. Files are not downloaded if they already exist in the cache.')
    parser.add_argument('--job', type=str, required=True, help='Name of the preprocessing job to run.')
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
    os.makedirs(os.path.join(args.cache_dir, args.dest_prefix), exist_ok=True)

    # Instantiate the job
    job_module = importlib.import_module(args.job)
    job = job_module.Job()

    # Ensure {cache-dir}/{source-prefix} is either empty or matches the contents
    # of {bucket}/{source-prefix}
    logging.info(f"Syncing s3://{args.bucket}/{args.source_prefix} to {args.cache_dir}/{args.source_prefix}")
    for obj in bucket.objects.filter(Prefix=args.source_prefix):
        local_path = os.path.join(args.cache_dir, obj.key)
        if not os.path.exists(local_path):
            logging.info(f"Downloading missing file s3://{args.bucket}/{obj.key} to {local_path}")
            bucket.download_file(obj.key, local_path)

    logging.info(f"Running preprocessing job '{args.job}' on s3://{args.bucket}/{args.source_prefix} and uploading to s3://{args.bucket}/{args.dest_prefix}")

    # For all .parquet files in {cache-dir}/{source-prefix}
    files = glob.glob(os.path.join(args.cache_dir, args.source_prefix, "*.parquet"))
    files = sorted(files)
    for infile in files:
        outfile = os.path.join(args.cache_dir, args.dest_prefix, os.path.basename(infile))
        job.run(infile, outfile)
        if not os.path.exists(outfile):
            raise Exception(f"Preprocessing job did not produce the output file {outfile}")
        bucket.upload_file(outfile, os.path.join(args.dest_prefix, os.path.basename(outfile)))
        logging.info(f"Uploaded '{outfile}' to s3://{args.bucket}/{args.dest_prefix}/{os.path.basename(outfile)}")

    # Write the job summary as JSON to the cache dir and bucket
    json.dump(job.summary(), open(os.path.join(args.cache_dir, args.dest_prefix, "summary.json"), "w"))
    bucket.upload_file(os.path.join(args.cache_dir, args.dest_prefix, "summary.json"), os.path.join(args.dest_prefix, "summary.json"))
    logging.info(f"Uploaded '{os.path.join(args.cache_dir, args.dest_prefix, 'summary.json')}' to s3://{args.bucket}/{args.dest_prefix}/summary.json")
