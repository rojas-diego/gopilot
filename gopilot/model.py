# Implementation of a basic GPT model using PyTorch Lightning. The model uses
# sinuosoidal positional encodings, stacked decoder layers, and a linear output
# layer. Masking is used to prevent the model from attending to future tokens.

import math
import yaml

import torch
from torch import Tensor
from torch.nn import Embedding, TransformerDecoder, TransformerDecoderLayer, Linear, Module as TorchModule, CrossEntropyLoss
from dataclasses import dataclass

@dataclass
class GPTModelConfig:
    vocab_size: int
    context_length: int
    embedding_dimensions: int
    num_layers: int
    num_heads: int
    feedforward_dimensions: int


class GPTModel(TorchModule):
    def __init__(self, vocab_size: int, context_length: int, embedding_dimensions: int, num_layers: int, num_heads: int, feedforward_dimensions: int, dropout: float = 0.0):
        super().__init__()
        self.vocab_size = vocab_size
        self.context_length = context_length
        self.embedding_dimensions = embedding_dimensions
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.feedforward_dimensions = feedforward_dimensions

        self.embeddings = Embedding(vocab_size, embedding_dimensions)
        self.positional_encodings = SinusoidalPositionalEncoding(context_length, embedding_dimensions)
        self.transformer_decoder_layer = TransformerDecoderLayer(embedding_dimensions, num_heads, feedforward_dimensions, dropout, batch_first=True)
        self.transformer_decoder = TransformerDecoder(self.transformer_decoder_layer, num_layers)
        # We disable biases as per the "Cramming" paper:
        # > We find empirical gains from disabling all linear layer biases
        # > (Dayma et al., 2021). Just as for the attention layers, this
        # > leverages the scaling law by accelerating gradient com- putation 
        # > without noticeable impacts on model size
        self.linear = Linear(embedding_dimensions, vocab_size, bias=False)
        self.loss = CrossEntropyLoss()

    def forward(self, batch: Tensor):
        batch_size, sequence_length = batch.shape
        assert sequence_length == self.context_length, f'Expected input of shape (batch_size, {self.context_length}), got ({batch_size}, {sequence_length})'

        embeddings = self.embeddings(batch) * math.sqrt(self.embedding_dimensions)
        assert embeddings.shape == (batch_size, self.context_length, self.embedding_dimensions)

        embeddings = self.positional_encodings(embeddings)
        assert embeddings.shape == (batch_size, self.context_length, self.embedding_dimensions)

        attn_mask = torch.triu(torch.ones((embeddings.size(1), embeddings.size(1)), device=embeddings.device, dtype=torch.bool), diagonal=1) # (context_length, context_length)
        assert attn_mask.shape == (self.context_length, self.context_length)

        logits = self.transformer_decoder(embeddings, embeddings, tgt_mask=attn_mask)
        logits = self.linear(embeddings)

        return logits
    
    @classmethod
    def from_config_file(cls, config_file: str, **kwargs):
        with open(config_file) as f:
            config = yaml.safe_load(f)
        return cls(**config, **kwargs)
    
    def hyperparams(self) -> dict:
        return {
            'vocab_size': self.vocab_size,
            'context_length': self.context_length,
            'embedding_dimensions': self.embedding_dimensions,
            'num_layers': self.num_layers,
            'num_heads': self.num_heads,
            'feedforward_dimensions': self.feedforward_dimensions
        }


class SinusoidalPositionalEncoding(TorchModule):
    def __init__(self, context_length: int, embedding_dimensions: int):
        super().__init__()
        self.context_length = context_length
        self.embedding_dimensions = embedding_dimensions
        self.register_buffer('positional_encoding', self.make_positional_encodings())

    def make_positional_encodings(self):
        pe = torch.zeros(self.context_length, self.embedding_dimensions)
        position = torch.arange(0, self.context_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, self.embedding_dimensions, 2).float() * -(math.log(10000.0) / self.embedding_dimensions))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        return pe

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.positional_encoding
