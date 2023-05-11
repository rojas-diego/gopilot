import sys
import os

def tokenize(filename):
    output = os.popen(f'go run tokenizer/scanner/scanner.go < {filename}').read()
    return output

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('Specify a file to tokenize.')
    else:
        print(tokenize(sys.argv[1]))
