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
go_scanner_scan_symbol.restype = ctypes.c_char_p

go_scanner_id_to_token_name_symbol = GOTOK_HANDLE.IDToTokenName
go_scanner_id_to_token_name_symbol.argtypes = [ctypes.c_int]
go_scanner_id_to_token_name_symbol.restype = ctypes.c_char_p

go_scanner_id_to_token_literal_symbol = GOTOK_HANDLE.IDToTokenLiteral
go_scanner_id_to_token_literal_symbol.argtypes = [ctypes.c_int]
go_scanner_id_to_token_literal_symbol.restype = ctypes.c_char_p

go_scanner_free_cstring_symbol = GOTOK_HANDLE.FreeCString
go_scanner_free_cstring_symbol.argtypes = [ctypes.c_char_p]

@dataclass
class ScanResult:
    offsets: List[List[int]]
    ids: List[int]
    names: List[str]
    literals: List[str]

def go_scanner_scan(src: str) -> ScanResult:
    try:
        stringp = go_scanner_scan_symbol(src.encode("utf-8"))
        scan_result_json = str(stringp, encoding="utf-8")
        result = ScanResult(**json.loads(scan_result_json))
        go_scanner_free_cstring_symbol(stringp)
        return result
    except Exception as e:
        raise Exception(f"Exception raised in libgotok.Scan(): {e}")

def go_scanner_id_to_token_name(id: int) -> str:
    try:
        stringp = go_scanner_id_to_token_name_symbol(id)
        result = str(stringp, encoding="utf-8")
        go_scanner_free_cstring_symbol(stringp)
        return result
    except Exception as e:
        raise Exception(f"Exception raised in libgotok.IDToTokenName(): {e}")

def go_scanner_id_to_token_literal(id: int) -> str:
    try:
        stringp = go_scanner_id_to_token_literal_symbol(id)
        result = str(stringp, encoding="utf-8")
        go_scanner_free_cstring_symbol(stringp)
        return result
    except Exception as e:
        raise Exception(f"Exception raised in libgotok.IDToTokenLiteral(): {e}")
