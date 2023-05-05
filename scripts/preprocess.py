# This script is used to download the data from v1.2 of the `Stack` dataset
# specific to go and apply multiple preprocessing steps to it.
#
# Make sure you are authenticated with the HuggingFace API by running
# `huggingface-cli login` and that you have requested access to the dataset on
# the HuggingFace website. Otherwise, an error will be thrown.
#
# The preprocessing steps are:
# - (Optional) Remove all comments
# - Remove code below a certain alphanumeric fraction
#
# It outputs a transformed version of the original dataset that can be used for
# training. It also selects high quality code samples randomly and dumps them to
# a single file that can be used to train the tokenizer.
#
# Here is some general information on the dataset: Number of samples: 4730461
# Alphanumeric fraction: 0.68 (0.57, 0.78) Size: 5440.98 (215.00, 15592.00)
# Average line length: 33.60 (14.14, 39.63) Max line length: 278.23 (43.00,
# 198.00) Stars: 261.11 (0.00, 365.00)
#
# See: https://huggingface.co/datasets/bigcode/the-stack-dedup/tree/main/data/go

import argparse
import os
import random
from datasets import load_dataset
import tqdm


PP_DEFAULT_CACHE_DIR = "data/raw"
PP_DEFAULT_CODE_OUTPUT_DIR = "data/processed"
PP_DEFAULT_TOKENIZER_OUTPUT_FILE = "data/tokenizer_dataset.go"

PP_ALPHA_NUMERIC_FRACTION_THRESHOLD = 0.6
PP_CHUNK_SIZE_BYTES = 200_000_000 # 200 MB

def remove_comments(content):
    """Removes lines that start with `//`."""
    lines = content.split("\n")
    lines = [line for line in lines if not line.startswith("//")]
    return "\n".join(lines)

def normalize_sample(content):
    """Removes all weird characters like Emojis and Chinese characters."""
    return content.encode("ascii", "ignore").decode("ascii")

def preprocess_sample(sample, args):
    content = normalize_sample(sample["content"])
    if args.remove_comments:
        content = remove_comments(content)
    return content


def sample_should_be_used_for_training_tokenizer(sample, args):
    # Less than 10% of the samples are used to train the tokenizer.
    if random.random() > 0.1:
        return False
    if sample["alphanum_fraction"] < PP_ALPHA_NUMERIC_FRACTION_THRESHOLD:
        return False
    if sample["size"] < 100 or sample["size"] > 10000:
        return False
    if sample["avg_line_length"] < 15 or sample["avg_line_length"] > 50:
        return False
    return True


def sample_should_be_used_in_training(sample, args):
    if sample["alphanum_fraction"] < PP_ALPHA_NUMERIC_FRACTION_THRESHOLD:
        return False
    if sample["size"] < 100 or sample["size"] > 10000:
        return False
    if sample["avg_line_length"] < 15 or sample["avg_line_length"] > 50:
        return False
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Preprocess the Stack dataset.')
    parser.add_argument('--remove-comments', action='store_true', help='Remove all comments from the dataset.')
    parser.add_argument('--cache-dir', type=str, default=PP_DEFAULT_CACHE_DIR, help='(Optional) The directory to read the raw dataset from. If not found, it is downloaded.')
    parser.add_argument('--code-ds-output-dir', type=str, default=PP_DEFAULT_CODE_OUTPUT_DIR, help='(Optional) The directory to output the processed dataset chunks to.')
    parser.add_argument('--tokenizer-ds-output-file', type=str, default=PP_DEFAULT_TOKENIZER_OUTPUT_FILE, help='(Optional) The file to output the tokenizer dataset to.')
    args = parser.parse_args()

    dedup_stack_go_dataset = load_dataset("bigcode/the-stack-dedup", cache_dir=args.cache_dir, data_dir="data/go", split="train", use_auth_token=True)

    with open(args.tokenizer_ds_output_file, "w") as tokenizer_ds_file:
        num_chunks = 1
        current_chunk_size = 0
        current_chunk_file = open(f"{args.code_ds_output_dir}/chunk_0.go", "w")

        progress_bar = tqdm.tqdm(enumerate(dedup_stack_go_dataset), desc="Processing dataset", mininterval=0.5)
        for i, sample in progress_bar:
            processed_sample = preprocess_sample(sample, args)
            
            if sample_should_be_used_in_training(sample, args):
                current_chunk_file.write(processed_sample)
                current_chunk_size += len(processed_sample)
                if current_chunk_size > PP_CHUNK_SIZE_BYTES:
                    current_chunk_file.close()
                    current_chunk_file = open(f"{args.code_ds_output_dir}/chunk_{num_chunks}.go", "w")
                    current_chunk_size = 0
                    num_chunks += 1
                else:
                    current_chunk_file.write("\n---\n")

            if sample_should_be_used_for_training_tokenizer(sample, args):
                tokenizer_ds_file.write(processed_sample)
                tokenizer_ds_file.write("\n---\n")

            progress_bar.set_postfix_str(f"Building Chunk: {num_chunks}", refresh=False)

            if num_chunks > 5:
                break
            
        current_chunk_file.close()

    # Rename all the chunks to match the format `{chunk_id}_out_of_{num_chunks}.go`}` 
    for i in range(num_chunks):
        os.rename(f"{args.code_ds_output_dir}/chunk_{i}.go", f"{args.code_ds_output_dir}/{i}_out_of_{num_chunks}.go")
