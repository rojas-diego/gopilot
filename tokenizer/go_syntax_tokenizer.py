import sys
import os
import json
from pdb import set_trace as st

def tokenize_file(filename):
    output = os.popen(f'go run tokenizer/scanner/scanner.go < {filename}').read()
    output = json.loads(output)
    output = list(zip(output['token_ids'], output['token_names'], output['token_values']))
    return output

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('Specify a file to tokenize.')
    else:
        print(tokenize_file(sys.argv[1]))
