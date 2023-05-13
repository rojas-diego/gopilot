import sys
import os
import json
from pdb import set_trace as st
from tokenizer import go_scanner_scan

BASIC_TYPES = set(['int', 'int8', 'int16 ', 'int32 ', 'int64', 'uint', 'uint8', 'uint16', 'uint32', 'uint64', 'uintptr', 'float32', 'float64', 'complex64', 'complex128', 'string', 'bool', 'byte', 'rune',])

def get_string_literal(string):
    return string[1:-1]

def get_comment_literal(comment):
    if comment[:2] == '//':
        return comment[2:]
    else:
        return comment[1:-1]

def get_tokens_of_type(toks, tok_type): 
    return list(map(
        lambda type_tuple: type_tuple[1], 
        filter(
            lambda tok_tuple: tok_tuple[0] == tok_type,
            toks
        )
    ))

def tokenize_file(filename):
    with open(filename, 'r') as f:
        return tokenize_string(f.read())
    
def tokenize_string(src):
    go_scanner_results = go_scanner_scan(src)
    st()
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

    string_toks = list(map(get_string_literal, string_toks))
    comment_toks = list(map(get_comment_literal, comment_toks))
    
    combined_bpe_toks = string_toks + ident_toks + comment_toks + int_toks + float_toks
    tok_file_text = '[SEP]'.join(combined_bpe_toks)
    with open('temp_tokens', 'w') as f:
        f.write(tok_file_text)

    tokenizer = new_tokenizer()
    train_tokenizer(tokenizer, ['temp_tokens'], 2**15)
    # save_tokenizer(tokenizer, 'config/tokenizer_gs.json')

    new_tokens = []

    for token in tokens:
        if token[0] == 'STRING':
            # remove quotes then feed to tokenizer
            pass
        elif token[0] == 'COMMENT':
            pass
        elif token[0] in ['IDENT', 'INT', 'FLOAT']:
            pass
        else:
            new_tokens.append(token)

    return new_tokens

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('Specify a file to tokenize.')
    else:
        print(tokenize_file(sys.argv[1]))
