# Play around with the different tokenizers.

import argparse
import sys
import tokenizer

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help='Path to the tokenizer configuration file.')
    parser.add_argument('--name', type=str, required=True, help='Name of the tokenizer.')
    args = parser.parse_args()

    # Instantiate tokenizer.{name}
    tokenizer_class = getattr(tokenizer, args.name)
    tok: tokenizer.Tokenizer = tokenizer_class.from_file(args.config)

    print("Vocab Size:", tok.get_vocab_size())
    stdin_contents = sys.stdin.read()
    print("----- Input -----", stdin_contents, sep="\n")
    print("----- Result -----", tok.encode(stdin_contents), sep="\n")
    print("----- Decoded -----", [tok.id_to_token(id) for id in tok.encode(stdin_contents)], sep="\n")
    print("----- Reconstructed -----", tok.decode(tok.encode(stdin_contents)), sep="\n")
