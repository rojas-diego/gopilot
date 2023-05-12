from dataclasses import dataclass
import json
import sys
from typing import Any, List
import os
import ctypes

if sys.platform == "win32":
    lib_ext = ".dll"
else:
    lib_ext = ".so"

GOTOK_HANDLE = ctypes.cdll.LoadLibrary(os.path.join(os.path.dirname(__file__), "libgotok" + lib_ext))

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

class Tokenizer:
    def __init__(self):
        self._encode = GOTOK_HANDLE.Encode
        self._encode.restype = ctypes.c_char_p
        self._encode.argtypes = [ctypes.c_char_p, ctypes.c_char_p]
        self._decode = GOTOK_HANDLE.Decode
        self._decode.restype = ctypes.c_char_p
        self._decode.argtypes = [ctypes.c_char_p, ctypes.c_char_p]
        self.config = Config(
            vocab_size=0
        )

    def encode(self, sequence: str) -> EncodeResult:
        encode_args = EncodeArgs(sequence).__dict__
        encode_args_json = json.dumps(encode_args).encode("utf-8")
        config_json = self.to_json().encode("utf-8")
        encode_result_json = self._encode(config_json, encode_args_json)
        return json.loads(encode_result_json, object_hook=lambda d: EncodeResult(**d))
    
    def decode(self, ids: List[int]) -> str:
        decode_args = DecodeArgs(ids).__dict__
        decode_args_json = json.dumps(decode_args).encode("utf-8")
        config_json = self.to_json().encode("utf-8")
        decode_result_json = self._decode(config_json, decode_args_json)
        return str(decode_result_json, encoding="utf-8")

    def get_vocab_size(self) -> int:
        return self.config.vocab_size

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
    # Example usage
    tokenizer = Tokenizer()

    code = """package main

    import "fmt"

    func main() {
        fmt.Println("Hello, World!")
    }
    """

    encode_result = tokenizer.encode(code)
    print(encode_result)

    decode_result = tokenizer.decode(encode_result.ids)
    print(decode_result)
