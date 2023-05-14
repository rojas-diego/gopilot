import json
from transformers import GPTBigCodeConfig, GPTBigCodeForCausalLM
from dataclasses import dataclass

import yaml

@dataclass
class GopilotConfig:
    vocab_size: int
    context_length: int
    embedding_dimensions: int
    num_layers: int
    num_heads: int
    feedforward_dimensions: int

class GopilotModel(GPTBigCodeForCausalLM):
    def __init__(self, vocab_size: int, context_length: int, embedding_dimensions: int, num_layers: int, num_heads: int, feedforward_dimensions: int):
        super().__init__(
            config=GPTBigCodeConfig(
                vocab_size=vocab_size,
                n_embd=embedding_dimensions,
                n_layer=num_layers,
                n_head=num_heads,
                n_inner=feedforward_dimensions,
                n_positions=context_length,
                eos_token_id=-1,
                bos_token_id=-1,
            )
        )

    @classmethod
    def from_config_file(cls, path: str, dropout: float = 0.0):
        return cls(**yaml.safe_load(open(path, "r")))
