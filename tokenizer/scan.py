import ctypes
from dataclasses import dataclass
import json
import os
import sys
from typing import List

if sys.platform == "win32":
    lib_ext = ".dll"
else:
    lib_ext = ".so"

GOTOK_HANDLE = ctypes.cdll.LoadLibrary(os.path.join(os.path.dirname(__file__), "libgotok" + lib_ext))

go_scanner_scan_symbol = GOTOK_HANDLE.Scan
go_scanner_scan_symbol.argtypes = [ctypes.c_char_p]
go_scanner_scan_symbol.restype = ctypes.c_void_p

go_scanner_id_to_token_name_symbol = GOTOK_HANDLE.IDToTokenName
go_scanner_id_to_token_name_symbol.argtypes = [ctypes.c_int]
go_scanner_id_to_token_name_symbol.restype = ctypes.c_void_p

go_scanner_id_to_token_literal_symbol = GOTOK_HANDLE.IDToTokenLiteral
go_scanner_id_to_token_literal_symbol.argtypes = [ctypes.c_int]
go_scanner_id_to_token_literal_symbol.restype = ctypes.c_void_p

go_scanner_free_cstring_symbol = GOTOK_HANDLE.FreeCString
go_scanner_free_cstring_symbol.argtypes = [ctypes.c_void_p]

@dataclass
class ScanResult:
    offsets: List[List[int]]
    ids: List[int]
    names: List[str]
    literals: List[str]

def go_scanner_scan(src: str) -> ScanResult:
    try:
        str_pointer = go_scanner_scan_symbol(src.encode("utf-8"))
        str_value = str(ctypes.cast(str_pointer, ctypes.c_char_p).value, encoding="utf-8") # type: ignore
        result = ScanResult(**json.loads(str_value))
        go_scanner_free_cstring_symbol(str_pointer)
        return result
    except Exception as e:
        raise Exception(f"Exception raised in libgotok.Scan(): {e}")

def go_scanner_id_to_token_name(id: int) -> str:
    try:
        str_pointer = go_scanner_id_to_token_name_symbol(id)
        str_value = str(ctypes.cast(str_pointer, ctypes.c_char_p).value, encoding="utf-8") # type: ignore
        go_scanner_free_cstring_symbol(str_pointer)
        return str_value
    except Exception as e:
        raise Exception(f"Exception raised in libgotok.IDToTokenName(): {e}")

def go_scanner_id_to_token_literal(id: int) -> str:
    try:
        str_pointer = go_scanner_id_to_token_literal_symbol(id)
        str_value = str(ctypes.cast(str_pointer, ctypes.c_char_p).value, encoding="utf-8") # type: ignore
        go_scanner_free_cstring_symbol(str_pointer)
        return str_value
    except Exception as e:
        raise Exception(f"Exception raised in libgotok.IDToTokenLiteral(): {e}")
