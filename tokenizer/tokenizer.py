from abc import ABC, abstractmethod
from typing import Iterable, List
from .scan import go_scanner_scan
from tokenizers import Tokenizer as WrappedHFTokenizer
from tokenizers.trainers import Trainer as HFTrainer

class Tokenizer(ABC):
    @abstractmethod
    def encode(self, sequence: str) -> List[int]:
        raise NotImplementedError
    
    @abstractmethod
    def decode(self, ids: List[int]) -> str:
        pass

    @abstractmethod
    def get_vocab_size(self) -> int:
        pass

    @abstractmethod
    def train_from_iterator(self, itertor: Iterable, trainer: HFTrainer):
        pass

    @abstractmethod
    def token_to_id(self, token: str) -> int:
        pass

    @abstractmethod
    def id_to_token(self, id: int) -> str:
        pass

    @abstractmethod
    def save(self, path: str):
        pass

    @abstractmethod
    @classmethod
    def from_file(cls, path: str) -> "Tokenizer":
        pass


class HFTokenizer(Tokenizer):
    def __init__(self, tokenizer: WrappedHFTokenizer):
        self.tokenizer = tokenizer


class GoScannerTokeniezr(Tokenizer):
    def __init__(self):
        pass


class GoAstTokenizer(Tokenizer):
    def __init__(self):
        pass
