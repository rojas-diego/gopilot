import sys
import os
import json
from pdb import set_trace as st

BASIC_TYPES = set(['int', 'int8', 'int16 ', 'int32 ', 'int64', 'uint', 'uint8', 'uint16', 'uint32', 'uint64', 'uintptr', 'float32', 'float64', 'complex64', 'complex128', 'string', 'bool', 'byte', 'rune',])

def get_tokens_of_type(toks, tok_type): 
    return list(map(
        lambda type_tuple: type_tuple[1], 
        filter(
            lambda tok_tuple: tok_tuple[0] == tok_type,
            toks
        )
    ))

def tokenize_file(filename):
    go_scanner_results = os.popen(f'go run tokenizer/scanner/scanner.go < {filename}').read()
    token_json = json.loads(go_scanner_results)
    
    tokens = list(
        map(
            lambda tok_tuple: ('IDENT_TYPE', tok_tuple[1]) if tok_tuple[1] in BASIC_TYPES else tok_tuple,
            zip(token_json['token_names'], token_json['token_values'])
        )
    )
    
    string_toks = get_tokens_of_type(tokens, 'STRING')
    ident_toks = get_tokens_of_type(tokens, 'IDENT')
    comment_toks = get_tokens_of_type(tokens, 'COMMENT')
    int_toks = get_tokens_of_type(tokens, 'INT')
    float_toks = get_tokens_of_type(tokens, 'FLOAT')

    return tokens

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('Specify a file to tokenize.')
    else:
        print(tokenize_file(sys.argv[1]))
