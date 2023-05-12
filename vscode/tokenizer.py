# Implements a Go specific tokenizer based on the HuggingFace tokenizers
# library.
#
# The Tokenizer is a BPE learned tokenizer on Go code and configured using
# additional tokens.
#
# Special tokens:
# - [UNK]: Unknown token.
# - [PAD]: Padding token.
# - [CLS]: Start of a sequence.
# - [SEP]: Delimits multiple sequences.
#
# See: https://huggingface.co/docs/tokenizers/quicktour

import logging
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Metaspace
from tokenizers.processors import TemplateProcessing


def new_tokenizer():
    """Initializes a new tokenizer from the specified configuration."""
    tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
    tokenizer.pre_tokenizer = Metaspace() # type: ignore
    return tokenizer


def train_tokenizer(tokenizer: Tokenizer, files: list[str], vocab_size: int = 2**15):
    """Trains a new tokenizer on the specified dataset using BPE."""
    trainer = BpeTrainer(vocab_size=vocab_size, special_tokens=["[UNK]", "[PAD]", "[CLS]", "[SEP]"]) # type: ignore
    logging.info(f"Training tokenizer on {len(files)} files with vocab_size={vocab_size}...")
    tokenizer.train(files, trainer)
    tokenizer.post_processor = TemplateProcessing(
        single="[CLS] $A",
        special_tokens=[
            ("[CLS]", tokenizer.token_to_id("[CLS]")),
        ],
    ) # type: ignore


def save_tokenizer(tokenizer: Tokenizer, path: str):
    tokenizer.save(path)
    logging.info("Saved tokenizer to '%s'", path)


def load_tokenizer(path: str) -> Tokenizer:
    logging.info("Loading tokenizer from '%s'", path)
    return Tokenizer.from_file(path)

def metaspace_cleanup(text: str) -> str:
    return text.replace(" ", "").replace("▁", " ")
