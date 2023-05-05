# This script trains a learned BPE tokenizer on the provided dataset. The
# tokenizer is saved to the provided output directory.

import argparse
import logging
import os
import sys

# Add the parent directory of this script to the module search path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import gopilot.tokenizer as gptok

TT_DEFAULT_OUTPUT_PATH = "config/tokenizer.json"

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train a BPE tokenizer on the provided files.')
    parser.add_argument('--output-path', type=str, default=TT_DEFAULT_OUTPUT_PATH, help='The filepath the trained tokenizer should be saved to.')
    parser.add_argument('files', nargs='*', help='The raw code files to use for training the tokenizer.')
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')

    if not args.files:
        logging.fatal("No files provided for training the tokenizer.")
        sys.exit(1)

    tokenizer = gptok.new_tokenizer()
    gptok.train_tokenizer(tokenizer, args.files)
    gptok.save_tokenizer(tokenizer, args.output_path)
