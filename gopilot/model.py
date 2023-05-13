from transformers import GPTBigCodeModel, GPTBigCodeConfig, GPTBigCodeForCausalLM
from dataclasses import dataclass

@dataclass
class Config:
    vocab_size: int
    context_length: int
    embedding_dimensions: int
    num_layers: int
    num_heads: int
    feedforward_dimensions: int

class Model(GPTBigCodeModel):
    def __init__(self, vocab_size, context_length, embedding_dimensions, num_layers, num_heads, feedforward_dimensions):
        super().__init__(
            config=GPTBigCodeConfig(
                vocab_size=vocab_size,
                n_embd=embedding_dimensions,
                n_layer=num_layers,
                n_head=num_heads,
                n_inner=feedforward_dimensions,
                n_positions=context_length,
            )
        )
