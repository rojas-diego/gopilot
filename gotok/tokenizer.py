from dataclasses import dataclass
import json
import sys
from typing import List
import os
import ctypes

if sys.platform == "win32":
    lib_ext = ".dll"
else:
    lib_ext = ".so"

GOTOK_HANDLE = ctypes.cdll.LoadLibrary(os.path.join(os.path.dirname(__file__), "libgotok" + lib_ext))

@dataclass
class EncodeResult:
    tokens: List[int]

@dataclass
class DecodeResult:
    code: str

class Tokenizer:
    def __init__(self):
        self._encode = GOTOK_HANDLE.Encode
        self._encode.restype = ctypes.c_char_p
        self._encode.argtypes = [ctypes.c_char_p]
        self._decode = GOTOK_HANDLE.Decode
        self._decode.restype = ctypes.c_char_p
        self._decode.argtypes = [ctypes.c_char_p]

    def encode(self, code: str) -> EncodeResult:
        encode_args = {"code": code}
        encode_args_json = json.dumps(encode_args).encode("utf-8")
        encode_result_json = self._encode(encode_args_json)
        return json.loads(encode_result_json, object_hook=lambda d: EncodeResult(**d))
    
    def decode(self, tokens: List[int]) -> DecodeResult:
        decode_args = {"tokens": tokens}
        decode_args_json = json.dumps(decode_args).encode("utf-8")
        decode_result_json = self._decode(decode_args_json)
        return json.loads(decode_result_json, object_hook=lambda d: DecodeResult(**d))

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

decode_result = tokenizer.decode(encode_result.tokens)
print(decode_result)
