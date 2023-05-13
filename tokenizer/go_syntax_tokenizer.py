import sys
from pdb import set_trace as st
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Metaspace
from tokenizers.processors import TemplateProcessing
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

def tokenize_file(filename, tokenizer=None):
    with open(filename, 'r') as f:
        return tokenize_src_string(f.read(), tokenizer)
    
def tokenize_src_string(src, tokenizer=None):
    go_scanner_results = go_scanner_scan(src)
    
    tokens = list(
        map(
            lambda tok_tuple: ('IDENT_TYPE', tok_tuple[1]) if tok_tuple[1] in BASIC_TYPES else tok_tuple,
            zip(go_scanner_results.names, go_scanner_results.literals)
        )
    )

    string_toks = get_tokens_of_type(tokens, 'STRING')
    comment_toks = get_tokens_of_type(tokens, 'COMMENT')

    string_toks = list(map(get_string_literal, string_toks))
    comment_toks = list(map(get_comment_literal, comment_toks))

    combined_bpe_toks = string_toks + comment_toks
    
    if tokenizer == None:
        tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
        tokenizer.pre_tokenizer = Metaspace() # type: ignore
        trainer = BpeTrainer(vocab_size=2**15, special_tokens=["[UNK]", "[PAD]", "[CLS]", "[SEP]"]) # type: ignore
        tokenizer.train_from_iterator(combined_bpe_toks, trainer)
        tokenizer.post_processor = TemplateProcessing(
            single="[CLS] $A",
            special_tokens=[
                ("[CLS]", tokenizer.token_to_id("[CLS]")),
            ],
        ) # type: ignore

    new_tokens = []

    for token in tokens:
        if token[0] == 'STRING':
            new_tokens.append(('STRING_START', token[1][0]))
            new_tokens += [('STRING_FRAG', sf) for sf in tokenizer.encode(token[1][1:-1]).tokens[1:]]
            new_tokens.append(('STRING_END', token[1][-1]))
        elif token[0] == 'COMMENT':
            if token[1][:2] == '//':
                new_tokens.append(('INLINE_COMMENT_START', '//'))
                new_tokens += [('COMMENT_FRAG', sf) for sf in tokenizer.encode(token[1][2:]).tokens[1:]]
            else:
                new_tokens.append(('BLOCK_COMMENT_START', '/*'))
                new_tokens += [('COMMENT_FRAG', sf) for sf in tokenizer.encode(token[1][2:-2]).tokens[1:]]
                new_tokens.append(('BLOCK_COMMENT_END', '*/'))
        else:
            new_tokens.append(token)

    return new_tokens

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('Specify a file to tokenize.')
    else:
        print(tokenize_file(sys.argv[1]))
