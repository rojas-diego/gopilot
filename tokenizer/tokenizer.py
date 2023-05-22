from abc import ABC, abstractmethod
from typing import Iterable, List

from tokenizers import Tokenizer as _HuggingFaceTokenizer
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import ByteLevel as ByteLevelEncoder, Metaspace as MetaspaceEncoder
from tokenizers.decoders import ByteLevel as ByteLevelDecoder, Metaspace as MetaspaceDecoder
from tokenizers.trainers import BpeTrainer

from .scan import go_scanner_id_to_token_name, go_scanner_scan, go_scanner_id_to_token_literal

class Trainer:
    def __init__(self, tokenizer: _HuggingFaceTokenizer, vocab_size: int, special_tokens: List[str]):
        assert len(special_tokens) != 0
        self.tokenizer = tokenizer
        self.trainer = BpeTrainer(vocab_size=vocab_size, special_tokens=["[UNK]", "[PAD]", "[EOS]"]) # type: ignore

    def train_from_iterator(self, iterator: Iterable):
        return self.tokenizer.train_from_iterator(iterator, trainer=self.trainer)
    

class GopilotTrainer(Trainer):
    def train_from_iterator(self, iterator: Iterable):
        # First scan the iterator to get the Go tokens
        accumulated_sequences = []
        for sequence in iterator:
            scan_result = go_scanner_scan(sequence)
            # Only train on comments, strings, and identifiers
            accumulated_sequences.extend([scan_result.literals[i] for i in range(len(scan_result.names)) if scan_result.names[i] in ['IDENT', 'COMMENT', 'STRING', 'INT', 'FLOAT', 'IMAG', 'CHAR']])
        super().train_from_iterator(accumulated_sequences)


class Tokenizer(ABC):
    @abstractmethod
    def encode(self, sequence: str) -> List[int]:
        pass

    @abstractmethod
    def encode_batch(self, sequences: Iterable[str]) -> List[List[int]]:
        pass

    @abstractmethod
    def decode(self, ids: List[int]) -> str:
        pass

    @abstractmethod
    def new_trainer(self, vocab_size: int, special_tokens: List[str]) -> Trainer:
        pass

    @abstractmethod
    def get_vocab_size(self) -> int:
        pass

    @abstractmethod
    def special_token_to_id(self, token: str) -> int:
        pass

    @abstractmethod
    def id_to_token(self, id: int) -> str:
        pass

    @abstractmethod
    def save(self, path: str):
        pass


class HuggingFaceTokenizer(Tokenizer):
    def __init__(self):
        self.tokenizer = _HuggingFaceTokenizer(model=BPE(unk_token="[UNK]", fuse_unk=True)) # type: ignore
        self.tokenizer.pre_tokenizer = MetaspaceEncoder() # type: ignore
        self.tokenizer.decoder = MetaspaceDecoder() # type: ignore

    def new_trainer(self, vocab_size: int, special_tokens: List[str]) -> Trainer:
        return Trainer(self.tokenizer, vocab_size, special_tokens)

    def encode(self, sequence: str) -> List[int]:
        return self.tokenizer.encode(sequence).ids
    
    def encode_batch(self, sequences: Iterable[str]) -> List[List[int]]:
        result = self.tokenizer.encode_batch(sequences)
        return [r.ids for r in result]

    def decode(self, ids: List[int]) -> str:
        return self.tokenizer.decode(ids)

    def get_vocab_size(self) -> int:
        return self.tokenizer.get_vocab_size()

    def special_token_to_id(self, token: str) -> int:
        return self.tokenizer.token_to_id(token)

    def id_to_token(self, id: int) -> str:
        return self.tokenizer.id_to_token(id)

    def save(self, path: str):
        self.tokenizer.save(path)

    @classmethod
    def from_file(cls, path: str) -> "Tokenizer":
        tokenizer = HuggingFaceTokenizer()
        tokenizer.tokenizer = _HuggingFaceTokenizer.from_file(path) # type: ignore
        return tokenizer


class GopilotTokenizer(Tokenizer):
    GO_TOKENS_COUNT = 89 + 3 # 89 Go tokens + 3 special tokens
    GO_SPACE_ID = 89
    GO_TOKENS_WITH_SPACE_AFTER = set([':', ',', ';', 'break', 'default', 'func', 'interface', 'select', 'case', 'defer', 'go', 'map', 'struct', 'chan', 'else', 'goto', 'package', 'switch', 'const', 'fallthrough', 'if', 'range', 'type', 'continue', 'for', 'import', 'return', 'var'])
    GO_TOKENS_WITH_SPACE_BEFORE_AFTER = set([':=', '=', '&&', '||', '>=', '<=', '>', '<', '!=', '==', '-', '+', '*', '/', '%', '+=', '-=', '*=', '/=', '%=', '<<', '>>'])
    GO_TOKENS_TO_REMOVE_SPACES_AFTER = GO_TOKENS_WITH_SPACE_AFTER.union(GO_TOKENS_WITH_SPACE_BEFORE_AFTER)

    def __init__(self):
        self.tokenizer = _HuggingFaceTokenizer(model=BPE(unk_token="[UNK]", fuse_unk=True))
        self.tokenizer.pre_tokenizer = ByteLevelEncoder() # type: ignore
        self.tokenizer.decoder = ByteLevelDecoder() # type: ignore

    def encode(self, sequence: str) -> List[int]:
        space_id = self.GO_SPACE_ID + self.tokenizer.get_vocab_size()
        scan_result = go_scanner_scan(sequence)
        expanded_tokens = []
        removing_spaces_after = False
        for i, token_name in enumerate(scan_result.names):
            if removing_spaces_after and token_name == 'SPACE':
                continue
            if token_name in self.GO_TOKENS_TO_REMOVE_SPACES_AFTER or token_name == 'SPACE':
                removing_spaces_after = True
            elif token_name != 'SPACE':
                removing_spaces_after = False
            if token_name in self.GO_TOKENS_WITH_SPACE_BEFORE_AFTER:
                while expanded_tokens[-1] == space_id:
                    expanded_tokens = expanded_tokens[:-1]
            # Expanded tokens
            if token_name in ['IDENT', 'COMMENT', 'STRING', 'INT', 'FLOAT', 'IMAG', 'CHAR']:
                tokens = self.tokenizer.encode(scan_result.literals[i])
                expanded_tokens.extend(tokens.ids)
            # Ignored tokens
            elif token_name in ['EOF']:
                continue
            # Go tokens
            else:
                expanded_tokens.append(scan_result.ids[i] + self.tokenizer.get_vocab_size())
        return expanded_tokens
    
    def decode(self, ids: List[int]) -> str:
        decoded = ""
        sequence_of_hf_tokens = []
        for id in ids:
            if id < self.tokenizer.get_vocab_size():
                sequence_of_hf_tokens.append(id)
            else:
                hf_decoded = self.tokenizer.decode(sequence_of_hf_tokens)
                decoded += hf_decoded[1:]
                sequence_of_hf_tokens = []
                token_literal = go_scanner_id_to_token_literal(id - self.tokenizer.get_vocab_size())
                if token_literal in self.GO_TOKENS_WITH_SPACE_BEFORE_AFTER:
                    decoded += ' '
                decoded += token_literal
                if token_literal in self.GO_TOKENS_TO_REMOVE_SPACES_AFTER:
                    decoded += ' '
        return decoded

    def id_to_token(self, id: int) -> str:
        if id < self.tokenizer.get_vocab_size():
            return self.tokenizer.decode([id])[1:]
        return go_scanner_id_to_token_literal(id - self.tokenizer.get_vocab_size())

    def id_to_token_name(self, id: int) -> str:
        if id < self.tokenizer.get_vocab_size():
            return self.tokenizer.id_to_token(id)
        token_name = go_scanner_id_to_token_name(id - self.tokenizer.get_vocab_size())
        if token_name in self.GO_TOKENS_WITH_SPACE_AFTER:
            token_name += '_'
        elif token_name in self.GO_TOKENS_WITH_SPACE_BEFORE_AFTER:
            token_name = '_' + token_name + '_'
        return token_name

    def encode_batch(self, sequences: Iterable[str]) -> List[List[int]]:
        return [self.encode(sequence) for sequence in sequences]

    def new_trainer(self, vocab_size: int, special_tokens: List[str]) -> Trainer:
        return GopilotTrainer(self.tokenizer, vocab_size - self.GO_TOKENS_COUNT, special_tokens)

    def special_token_to_id(self, token: str) -> int:
        return self.tokenizer.token_to_id(token)

    def get_vocab_size(self) -> int:
        return self.tokenizer.get_vocab_size() + self.GO_TOKENS_COUNT

    def save(self, path: str):
        self.tokenizer.save(path)

    @classmethod
    def from_file(cls, path: str) -> "GopilotTokenizer":
        tokenizer = cls()
        tokenizer.tokenizer = _HuggingFaceTokenizer.from_file(path) # type: ignore
        return tokenizer


class AdvancedGoScannerTokenizer(Tokenizer):
    NUM_GO_TOKENS = 89
    BASIC_TYPES = ['int', 'int8', 'int16 ', 'int32 ', 'int64', 'uint', 'uint8', 'uint16', 'uint32', 'uint64', 'uintptr', 'float32', 'float64', 'complex64', 'complex128', 'string', 'bool', 'byte', 'rune',]
    EXTRA_TOKENS = ['"', '//', '/*', '*/']

    def __init__(self):
        self.tokenizer = _HuggingFaceTokenizer(model=BPE(unk_token="[UNK]", fuse_unk=True)) # type: ignore
        self.tokenizer.pre_tokenizer = ByteLevelEncoder() # type: ignore
        self.tokenizer.decoder = ByteLevelDecoder() # type: ignore
        self.BASIC_TYPES_START = self.NUM_GO_TOKENS
        self.EXTRA_TOKENS_START = self.BASIC_TYPES_START + len(self.BASIC_TYPES)
        self.BPE_START = self.EXTRA_TOKENS_START + len(self.EXTRA_TOKENS)

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
                new_tokens += [sf + self.BPE_START for sf in self.tokenizer.encode(token[1][1:-1]).ids]
                new_tokens.append(self.EXTRA_TOKENS_START + self.EXTRA_TOKENS.index('"'))
            elif token[0] == 'COMMENT':
                if token[1][:2] == '//':
                    new_tokens.append(self.EXTRA_TOKENS_START + self.EXTRA_TOKENS.index('//'))
                    new_tokens += [sf + self.BPE_START for sf in self.tokenizer.encode(token[1][2:]).ids]
                else:
                    new_tokens.append(self.EXTRA_TOKENS_START + self.EXTRA_TOKENS.index('/*'))
                    new_tokens += [sf + self.BPE_START for sf in self.tokenizer.encode(token[1][2:-2]).ids]
                    new_tokens.append(self.EXTRA_TOKENS_START + self.EXTRA_TOKENS.index('*/'))
            elif token[0] == 'IDENT':
                new_tokens += [sf + self.BPE_START for sf in self.tokenizer.encode(token[1]).ids]
            elif token[0] == 'IDENT_TYPE':
                new_tokens.append(self.BASIC_TYPES_START + self.BASIC_TYPES.index(token[1]))
            else:
                new_tokens.append(token[2])

        return new_tokens

    def encode_batch(self, sequences: Iterable[str]) -> List[List[int]]:
        return [self.encode(sequence) for sequence in sequences]

    def decode(self, ids: List[int]) -> str:
        return ''.join([self.id_to_token(id) for id in ids])

    def id_to_token(self, id: int) -> str:
        if id < self.BASIC_TYPES_START:
            return go_scanner_id_to_token_name(id)
        elif id < self.EXTRA_TOKENS_START:
            return self.BASIC_TYPES[id - self.BASIC_TYPES_START]
        elif id < self.BPE_START:
            return self.EXTRA_TOKENS[id - self.EXTRA_TOKENS_START]
        else:
            return self.tokenizer.decode([id - self.BPE_START])
    
    def special_token_to_id(self, token: str) -> int:
        return self.tokenizer.token_to_id(token)

    def seq_to_bpe_tokens(self, seq):
        go_scanner_results = go_scanner_scan(seq)
        tokens = list(
            map(
                lambda tok_tuple: ('IDENT_TYPE', tok_tuple[1], tok_tuple[2]) if tok_tuple[1] in self.BASIC_TYPES else tok_tuple,
                zip(go_scanner_results.names, go_scanner_results.literals, go_scanner_results.ids)
            )
        )
        return [tok[1] for tok in tokens if tok[0] in ['IDENT', 'STRING', 'COMMENT']]

    def new_trainer(self, vocab_size: int, special_tokens: List[str]) -> Trainer:
        return Trainer(self.tokenizer, vocab_size - self.BPE_START, special_tokens)

    def get_vocab_size(self) -> int:
        return self.tokenizer.get_vocab_size() + self.BPE_START

    def save(self, path: str):
        self.tokenizer.save(path)

    @classmethod
    def from_file(cls, path: str) -> "Tokenizer":
        tokenizer = cls()
        tokenizer.tokenizer = _HuggingFaceTokenizer.from_file(path) # type: ignore
        return tokenizer
