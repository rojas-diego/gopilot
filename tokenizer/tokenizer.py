from dataclasses import dataclass
import json
import sys
from typing import Any, List
import os
import ctypes


@dataclass
class Config:
    vocab_size: int

@dataclass
class EncodeArgs:
    sequence: str

@dataclass
class EncodeResult:
    ids: List[int]
    offsets: List[int]
    attention_mask: List[int]
    special_tokens_mask: List[int]

@dataclass
class DecodeArgs:
    ids: List[int]


if sys.platform == "win32":
    lib_ext = ".dll"
else:
    lib_ext = ".so"

GOTOK_HANDLE = ctypes.cdll.LoadLibrary(os.path.join(os.path.dirname(__file__), "libgotok" + lib_ext))

go_scanner_scan_symbol = GOTOK_HANDLE.Scan
go_scanner_scan_symbol.argtypes = [ctypes.c_char_p]
go_scanner_scan_symbol.restype = ctypes.c_char_p

@dataclass
class ScanResult:
    offsets: List[List[int]]
    ids: List[int]
    names: List[str]
    literals: List[str]

def go_scanner_scan(src: str) -> ScanResult:
    scan_result_json = str(go_scanner_scan_symbol(src.encode("utf-8")), encoding="utf-8")
    return ScanResult(**json.loads(scan_result_json))


go_parser_parse_file_symbol = GOTOK_HANDLE.Parse
go_parser_parse_file_symbol.argtypes = [ctypes.c_char_p]
go_parser_parse_file_symbol.restype = ctypes.c_char_p

def go_parser_parse_file(src: str) -> str:
    return str(go_parser_parse_file_symbol(src.encode("utf-8")), encoding="utf-8")


class Tokenizer:
    def __init__(self):
        self.config = Config(
            vocab_size=0
        )

    def encode(self, sequence: str) -> EncodeResult:
        raise NotImplementedError
    
    def decode(self, ids: List[int]) -> str:
        raise NotImplementedError

    def get_vocab_size(self) -> int:
        raise NotImplementedError

    def save(self, path: str):
        with open(path, "w") as f:
            f.write(self.to_json())

    def to_json(self) -> str:
        return json.dumps(self.config.__dict__)

    def __call__(self, *args: Any, **kwds: Any) -> Any:
        return self.encode(*args, **kwds)

    @classmethod
    def from_file(cls, path: str) -> "Tokenizer":
        with open(path, "r") as f:
            return cls.from_json(f.read())
        
    @classmethod
    def from_json(cls, json_str: str) -> "Tokenizer":
        tokenizer = cls()
        tokenizer.config = Config(**json.loads(json_str))
        return tokenizer


if __name__ == "__main__":
    src = """package main

    import "fmt"

    func main() {
        fmt.Println("Hello, World!")
    }
    """

    print(go_scanner_scan(src))
    print(go_parser_parse_file(src))
