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
    def __init__(self, config: GopilotConfig, dropout: float = 0.0):
        self._config = config
        super().__init__(
            config=GPTBigCodeConfig(
                vocab_size=config.vocab_size,
                n_embd=config.embedding_dimensions,
                n_layer=config.num_layers,
                n_head=config.num_heads,
                n_inner=config.feedforward_dimensions,
                n_positions=config.context_length,
                eos_token_id=-1,
                bos_token_id=-1,
            )
        )

    @classmethod
    def from_config_file(cls, path: str, dropout: float = 0.0):
        return cls(GopilotConfig(**yaml.safe_load(open(path, "r"))), dropout=dropout)

    def get_config(self) -> GopilotConfig:
        return self._config
