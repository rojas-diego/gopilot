"""
This script is used to generate the subset of the pre-training dataset by
randomly sampling `x` samples from one of the shards of the pre-training
dataset. The subset is used for finetuning the model on the pre-training dataset
as a regulazing term.

Each sample is presented to the console and the user is asked to input a
`y` or `n` to indicate whether the sample should be included in the subset or
not.

The samples are outputted in the format of a JSONL file in
`dataset/finetuning/pre-training-subset.jsonl`
"""

import argparse
import json
import pandas as pd

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-file", type=str, required=True, help="Path to the input file")
    parser.add_argument("--output-file", type=str, default="pre-training-subset.jsonl", help="Path to the output file")
    parser.add_argument("--num-samples", type=int, required=True, help="Number of samples to generate")
    args = parser.parse_args()

    # Open the shard
    df = pd.read_parquet(args.input_file)

    # While the number of samples is less than the number of samples required
    # keep sampling
    with open(args.output_file, "a") as f:
        while True:
            row = df.sample(n=1).iloc[0]

            # Clear the screen
            print("\033c")

            # Print the row to the console
            print(row["content"])

            # Ask the user whether to include the sample in the subset
            include = input("Include? (Enter (yes)/n): ")

            # If the user says yes, add the sample to the subset
            if include != "n":
                f.write(json.dumps({"sample": row["content"]}) + "\n")
                f.flush()
