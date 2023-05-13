from pdb import set_trace as st
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import Metaspace
from tokenizers.processors import TemplateProcessing
from abc import ABC, abstractmethod
from typing import Iterable, List
from .scan import go_scanner_scan, go_scanner_id_to_token_name
from tokenizers.pre_tokenizers import Whitespace
from tokenizers import Tokenizer as WrappedHFTokenizer
from tokenizers.trainers import BpeTrainer, Trainer as HFTrainer
from pickle import load, dump
from dataclasses import dataclass
import json
import sys
from typing import Any, List
import os
import ctypes

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
    def train_from_iterator(self, iterator: Iterable):
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
    def from_file(cls, path: str) -> "Tokenizer":
        pass

class HFTokenizer(Tokenizer):
    def __init__(self, tokenizer: WrappedHFTokenizer, vocab_size: int, special_tokens: list):
        self.tokenizer = tokenizer
        self.tokenizer.pre_tokenizer = Metaspace()
        self.vocab_size = vocab_size
        self.special_tokens = special_tokens
        self.trainer = BpeTrainer(vocab_size=vocab_size, special_tokens=special_tokens)

    def encode(self, sequence: str) -> List[int]:
        return self.tokenizer.encode(sequence).ids
    
    def decode(self, ids: List[int]) -> str:
        return self.tokenizer.decode(ids)

    def get_vocab_size(self) -> int:
        return self.tokenizer.get_vocab_size()

    def train_from_iterator(self, iterator: Iterable):
        return self.tokenizer.train_from_iterator(iterator, self.trainer)

    def token_to_id(self, token: str) -> int:
        return self.tokenizer.token_to_id(token)

    def id_to_token(self, id: int) -> str:
        return self.tokenizer.id_to_token(id)

    def save(self, path: str):
        self.tokenizer.save(path)
        with open(path + '_', 'wb') as f:
            dump((self.vocab_size, self.special_tokens), f)

    @classmethod
    def from_file(cls, path: str) -> "Tokenizer":
        tokenizer = WrappedHFTokenizer.from_file(path)
        with open(path + '_', 'rb') as f:
            vocab_size, special_tokens = load(f)
        return HFTokenizer(tokenizer=tokenizer, vocab_size=vocab_size, special_tokens=special_tokens)

class GoScannerTokenizer(Tokenizer):
    NUM_GO_TOKENS = 89
    BASIC_TYPES = ['int', 'int8', 'int16 ', 'int32 ', 'int64', 'uint', 'uint8', 'uint16', 'uint32', 'uint64', 'uintptr', 'float32', 'float64', 'complex64', 'complex128', 'string', 'bool', 'byte', 'rune',]
    EXTRA_TOKENS = ['"', '//', '/*', '*/']

    def __init__(self, tokenizer: WrappedHFTokenizer, vocab_size: int, special_tokens: list):
        self.tokenizer = tokenizer
        self.tokenizer.pre_tokenizer = Metaspace()
        self.vocab_size = vocab_size
        self.special_tokens = special_tokens
        self.trainer = BpeTrainer(vocab_size=vocab_size - self.NUM_GO_TOKENS - len(self.BASIC_TYPES) - len(self.EXTRA_TOKENS), special_tokens=special_tokens)

        self.BASIC_TYPES_START = self.NUM_GO_TOKENS
        self.EXTRA_TOKENS_START = self.BASIC_TYPES_START + len(self.BASIC_TYPES)
        self.SPECIAL_TOKENS_START = self.EXTRA_TOKENS_START + len(self.EXTRA_TOKENS)
        self.BPE_START = self.SPECIAL_TOKENS_START + len(special_tokens)


    def _get_string_literal(self, string):
        return string[1:-1]

    def _get_comment_literal(self, comment):
        if comment[:2] == '//':
            return comment[2:]
        else:
            return comment[1:-1]

    def _get_tokens_of_type(self, toks, tok_type): 
        return list(map(
            lambda type_tuple: type_tuple[1], 
            filter(
                lambda tok_tuple: tok_tuple[0] == tok_type,
                toks
            )
        ))

    def encode(self, sequence: str) -> List[int]:
        go_scanner_results = go_scanner_scan(sequence)
        
        tokens = list(
            map(
                lambda tok_tuple: ('IDENT_TYPE', tok_tuple[1], tok_tuple[2]) if tok_tuple[1] in self.BASIC_TYPES else tok_tuple,
                zip(go_scanner_results.names, go_scanner_results.literals, go_scanner_results.ids)
            )
        )

        new_tokens = []

        for token in tokens:
            if token[0] == 'STRING':
                new_tokens.append(self.EXTRA_TOKENS_START + self.EXTRA_TOKENS.index('"'))
                new_tokens += [sf + self.BPE_START for sf in self.tokenizer.encode(token[1][1:-1]).ids[1:]]
                new_tokens.append(self.EXTRA_TOKENS_START + self.EXTRA_TOKENS.index('"'))
            elif token[0] == 'COMMENT':
                if token[1][:2] == '//':
                    new_tokens.append(self.EXTRA_TOKENS_START + self.EXTRA_TOKENS.index('//'))
                    new_tokens += [sf + self.BPE_START for sf in self.tokenizer.encode(token[1][2:]).ids[1:]]
                else:
                    new_tokens.append(self.EXTRA_TOKENS_START + self.EXTRA_TOKENS.index('/*'))
                    new_tokens += [sf + self.BPE_START for sf in self.tokenizer.encode(token[1][2:-2]).ids[1:]]
                    new_tokens.append(self.EXTRA_TOKENS_START + self.EXTRA_TOKENS.index('*/'))
            elif token[0] == 'IDENT':
                new_tokens += [sf + self.BPE_START for sf in self.tokenizer.encode(token[1]).ids]
            elif token[0] == 'IDENT_TYPE':
                new_tokens.append(self.BASIC_TYPES_START + self.BASIC_TYPES.index(token[1]))
            else:
                new_tokens.append(token[2])

        return new_tokens

    def decode(self, ids: List[int]) -> str:
        return ''.join([self.id_to_token(id) for id in ids])

    def id_to_token(self, id: int) -> str:
        if id < self.BASIC_TYPES_START:
            return go_scanner_id_to_token_name(id)
        elif id < self.EXTRA_TOKENS_START:
            return self.BASIC_TYPES[id - self.BASIC_TYPES_START]
        elif id < self.SPECIAL_TOKENS_START:
            return self.EXTRA_TOKENS[id - self.EXTRA_TOKENS_START]
        elif id < self.BPE_START:
            return self.special_tokens[id - self.SPECIAL_TOKENS_START]
        else:
            return self.tokenizer.decode([id - self.BPE_START])
    
    def token_to_id(self, token: str) -> int:
        return self.tokenizer.token_to_id(token)

    def train_from_iterator(self, iterator: Iterable):
        
        return self.tokenizer.train_from_iterator(iterator, self.trainer)




    def get_vocab_size(self) -> int:
        return self.tokenizer.get_vocab_size() + self.NUM_GO_TOKENS

    def save(self, path: str):
        self.tokenizer.save(path)
        with open(path + '_', 'wb') as f:
            dump((self.vocab_size, self.special_tokens), f)

    @classmethod
    def from_file(cls, path: str) -> "Tokenizer":
        tokenizer = WrappedHFTokenizer.from_file(path)
        with open(path + '_', 'rb') as f:
            vocab_size, special_tokens = load(f)
        return GoScannerTokenizer(tokenizer=tokenizer, vocab_size=vocab_size, special_tokens=special_tokens)


class GoAstTokenizer(Tokenizer):
    def __init__(self, tokenizer: WrappedHFTokenizer, vocab_size: int, special_tokens: list):
        self.tokenizer = tokenizer

    def encode(self, sequence: str) -> List[int]:
        raise NotImplementedError
    
    def decode(self, ids: List[int]) -> str:
        pass

    def get_vocab_size(self) -> int:
        pass

    def train_from_iterator(self, iterator: Iterable):
        pass

    def token_to_id(self, token: str) -> int:
        pass

    def id_to_token(self, id: int) -> str:
        pass

    def save(self, path: str):
        pass

    @classmethod
    def from_file(cls, path: str) -> "Tokenizer":
        pass
