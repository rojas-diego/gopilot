import logging
import math
import os
from typing import Iterable

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from tokenizers import (Tokenizer, decoders, models, pre_tokenizers, trainers)
from tokenizers.processors import TemplateProcessing
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torchtext.datasets import WikiText2

import dlutils

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

EMBEDDING_DIMENSIONS = 100
BATCH_SIZE = 32
DATA_DIR = "./data"
MAX_SEQUENCE_LEN = 512
PAD_TOKEN_ID = 1
SEP_TOKEN_ID = 2
MODEL = "transformer"

# Hack to download the dataset bc the tokenizer needs it and it's lazily
# downloaded.
if not os.stat("data/datasets/WikiText2/wikitext-2/wiki.train.tokens"):
    train_datapipe, validation_datapipe, test_datapipe = WikiText2(split=('train', 'valid', 'test'), root='data')  # type: ignore
    for elem in zip(train_datapipe, validation_datapipe, test_datapipe):
        break


def train_tokenizer(training_file: str, output_dir: str, vocab_size: int):
    # If tokenizer already exists, do nothing
    if os.path.exists(os.path.join(output_dir, "wiki_tokenizer.json")):
        return

    os.makedirs(output_dir, exist_ok=True)

    # Initialize a BPE model
    bpe_model = models.BPE()

    # Initialize a pre-tokenizer (split text into words)
    pre_tokenizer = pre_tokenizers.Whitespace()

    # Initialize a decoder (reconstruct text from tokens)
    decoder = decoders.BPEDecoder()

    # Initialize a trainer (set training parameters)
    trainer = trainers.BpeTrainer(vocab_size=vocab_size, initial_alphabet=pre_tokenizers.ByteLevel.alphabet())

    # Combine all components into a tokenizer
    tokenizer = Tokenizer(bpe_model)
    tokenizer.pre_tokenizer = pre_tokenizer
    tokenizer.decoder = decoder
    tokenizer.post_processor = TemplateProcessing(
        single="$0",
        special_tokens=[
            ("$0", 0),
            ("[PAD]", PAD_TOKEN_ID),
            ("[SEP]", SEP_TOKEN_ID),
        ],
    )

    # Train the tokenizer
    tokenizer.train(files=[training_file], trainer=trainer)

    # Save the tokenizer
    tokenizer.save(os.path.join(output_dir, "wiki_tokenizer.json"))

train_tokenizer("data/datasets/WikiText2/wikitext-2/wiki.train.tokens", DATA_DIR, 10000)

train_datapipe, validation_datapipe, test_datapipe = WikiText2(split=('train', 'valid', 'test'), root='data')  # type: ignore

tokenizer: Tokenizer = Tokenizer.from_file(os.path.join(DATA_DIR, "wiki_tokenizer.json"))

logging.info("Vocabulary size: %d", tokenizer.get_vocab_size())

class WikiText2Dataset(Dataset):
    def __init__(self, dsiter: Iterable, max_sequence_len: int, pad_token_id: int, sep_token_id: int):
        self.samples = []
        concat_sequence = torch.tensor([], dtype=torch.long)

        for item in dsiter:
            tokens = torch.tensor(tokenizer.encode(item).ids, dtype=torch.long)

            # If the current sequence is too long, we continuously split it.
            while tokens.numel() != 0:
                if concat_sequence.numel() == 0:
                    split = min(max_sequence_len, tokens.numel())
                    concat_sequence = tokens[:split]
                    tokens = tokens[split:]
                elif concat_sequence.numel() + 1 == max_sequence_len:
                    concat_sequence = torch.cat([concat_sequence, torch.tensor([pad_token_id], dtype=torch.long)]) 
                else:
                    split = min(max_sequence_len - concat_sequence.numel() - 1, tokens.numel())
                    concat_sequence = torch.cat([concat_sequence, torch.tensor([sep_token_id], dtype=torch.long), tokens[:split]])
                    tokens = tokens[split:]

                # If the concat_sequence is full, we flush it
                if concat_sequence.numel() == max_sequence_len:
                    self.samples.append(concat_sequence)
                    concat_sequence = torch.tensor([], dtype=torch.long)

        # Pad the remaining tokens
        if concat_sequence.numel() != 0:
            if concat_sequence.numel() < max_sequence_len:
                concat_sequence = torch.cat([concat_sequence, torch.full((max_sequence_len - concat_sequence.numel(),), pad_token_id, dtype=torch.long)])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]

train_dataset = WikiText2Dataset(train_datapipe, MAX_SEQUENCE_LEN, PAD_TOKEN_ID, SEP_TOKEN_ID)
validation_dataset = WikiText2Dataset(validation_datapipe, MAX_SEQUENCE_LEN, PAD_TOKEN_ID, SEP_TOKEN_ID)
test_dataset = WikiText2Dataset(test_datapipe, MAX_SEQUENCE_LEN, PAD_TOKEN_ID, SEP_TOKEN_ID)

class LSTMLanguageModel(nn.Module):
    def __init__(self, vocab_size: int, embedding_dim: int, hidden_size: int, num_layers: int, dropout=0.5):
        super(LSTMLanguageModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_size, num_layers, dropout=dropout, batch_first=True)
        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, x, hidden):
        embedded = self.embedding(x)
        lstm_output, hidden = self.lstm(embedded, hidden)
        output = self.fc(lstm_output)
        return output, hidden

    def init_hidden(self, batch_size, device):
        return (torch.zeros(self.lstm.num_layers, batch_size, self.lstm.hidden_size).to(device),
                torch.zeros(self.lstm.num_layers, batch_size, self.lstm.hidden_size).to(device))

# Transformer implementation drawn from:
# https://pytorch.org/tutorials/beginner/transformer_tutorial.html
class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Arguments:
            x: Tensor, shape ``[seq_len, batch_size, embedding_dim]``
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)

# Transformer implementation drawn from:
# https://pytorch.org/tutorials/beginner/transformer_tutorial.html
class TransformerModel(nn.Module):
    def __init__(self, vocab_size: int, embedding_dims: int, num_heads: int, dim_feedforward: int, nlayers: int, dropout: float = 0.5):
        super().__init__()
        self.model_type = 'Transformer'
        self.pos_encoder = PositionalEncoding(embedding_dims, dropout)
        encoder_layers = TransformerEncoderLayer(embedding_dims, num_heads, dim_feedforward, dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
        self.encoder = nn.Embedding(vocab_size, embedding_dims)
        self.embedding_dims = embedding_dims
        self.decoder = nn.Linear(embedding_dims, vocab_size)
        self.init_weights()

    def init_weights(self) -> None:
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, src: torch.Tensor, src_mask: torch.Tensor) -> torch.Tensor:
        """
        Arguments:
            src: Tensor, shape ``[seq_len, batch_size]``
            src_mask: Tensor, shape ``[seq_len, seq_len]``

        Returns:
            output Tensor of shape ``[seq_len, batch_size, vocab_size]``
        """
        src = self.encoder(src) * math.sqrt(self.embedding_dims)
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src, src_mask)
        output = self.decoder(output)
        return output

# Transformer implementation drawn from:
# https://pytorch.org/tutorials/beginner/transformer_tutorial.html
def generate_square_subsequent_mask(sz: int) -> torch.Tensor:
    """Generates an upper-triangular matrix of ``-inf``, with zeros on ``diag``."""
    return torch.triu(torch.ones(sz, sz) * float('-inf'), diagonal=1)

if MODEL == "transformer":
    model = TransformerModel(tokenizer.get_vocab_size(), EMBEDDING_DIMENSIONS, 4, 2048, 3, 0.5)
else:
    model = LSTMLanguageModel(tokenizer.get_vocab_size(), EMBEDDING_DIMENSIONS, 128, 3)
    dlutils.xavier_initialization(model)

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=0.0001)

if MODEL == "transformer":
    task = dlutils.TransformerLanguageModeling(model, criterion, optimizer, generate_square_subsequent_mask(MAX_SEQUENCE_LEN))
else:
    task = dlutils.ReccurentLanguageModeling(model, criterion, optimizer)

trainer = dlutils.Trainer(
    task,
    dlutils.best_device(),
    train_loader=DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True),
    validation_loader=DataLoader(validation_dataset, batch_size=BATCH_SIZE, shuffle=False),
    test_loader=DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False),
)
trainer.register_handlers(
    dlutils.LoggingHandler(),
)
trainer.train(20)
trainer.test()
