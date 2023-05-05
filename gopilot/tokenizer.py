# Implements a Go specific tokenizer based on the HuggingFace tokenizers
# library.
#
# The Tokenizer is a BPE learned tokenizer on Go code and configured using
# additional tokens.
#
# Special tokens:
# - [SEP]: Delimits the start and end of a sequence.
# - [PAD]: Padding token.
# - [UNK]: Unknown token.
#
# See: https://huggingface.co/docs/tokenizers/quicktour

import logging
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.processors import TemplateProcessing


def new_tokenizer():
    """Initializes a new tokenizer from the specified configuration."""
    tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
    tokenizer.pre_tokenizer = Whitespace()
    return tokenizer


def train_tokenizer(tokenizer: Tokenizer, files: list[str]):
    """Trains a new tokenizer on the specified dataset using BPE."""
    trainer = BpeTrainer(special_tokens=["[UNK]", "[SEP]", "[PAD]"])
    logging.info("Training tokenizer on %d files...", len(files))
    tokenizer.train(files, trainer)
    tokenizer.post_processor = TemplateProcessing(
        single="[SEP] $A [SEP]",
        pair="[SEP] $A [SEP] $B [SEP]",
        special_tokens=[("[SEP]", tokenizer.token_to_id("[SEP]"))],
    )


def save_tokenizer(tokenizer: Tokenizer, path: str):
    tokenizer.save(path)
    logging.info("Saved tokenizer to '%s'", path)


def load_tokenizer(path: str):
    logging.info("Loading tokenizer from '%s'", path)
    return Tokenizer.from_file(path)
