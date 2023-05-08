# This script uses a trained model to generate predictions. You submit a Go file
# and the model will complete the code for you up until max_tokens tokens.

import argparse
import os
import sys
import torch

# Add the parent directory of this script to the module search path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import gopilot.tokenizer as gptok
import gopilot.model as gpmodel

def sample_with_temperature(logits, temperature):
    logits = logits / temperature
    probs = torch.softmax(logits, dim=-1)
    token_idx = torch.multinomial(probs, 1).item()
    return token_idx

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint-file', type=str, required=True, help='Path to the model checkpoint')
    parser.add_argument('--model', type=str, required=True, help='Path to the model configuration file.')
    parser.add_argument('--tokenizer', type=str, required=True, help='Path to the tokenizer configuration file.')
    parser.add_argument('--input-file', type=str, required=True, help='Path to the input file.')
    parser.add_argument('--max-tokens', type=int, default=8, help='Maximum number of tokens to generate.')
    parser.add_argument('--temperature', type=float, default=0.5, help='Sampling temperature.')
    args = parser.parse_args()

    # Load the model and tokenizer
    tokenizer = gptok.load_tokenizer(args.tokenizer)
    model = gpmodel.GPTModel.from_config_file(args.model)
    assert model.vocab_size == tokenizer.get_vocab_size(), "Model context length and tokenizer context length must match."

    # Load the checkpoint
    checkpoint = torch.load(args.checkpoint_file)
    model.load_state_dict(checkpoint['model'])

    # Execute the prompt
    with open(args.input_file, 'r') as f:
        input = f.read()
        tokens = tokenizer.encode(input).ids
        window = tokens[-model.context_length:]
        window.extend([tokenizer.token_to_id('[PAD]')] * (model.context_length - len(window)))
        window = torch.tensor(window) # (context_length,)
        print("--- INPUT ---")
        print(input)
        print("--- PREDICTIONS ---")
        for i in range(args.max_tokens):
            inputs = window.unsqueeze(0) # (1, context_length)
            output = model(inputs).detach() # (1, context_length, vocab_size)
            new_token = sample_with_temperature(output[0, -1, :], 0.5)
            window = torch.cat((window, torch.tensor([new_token], dtype=torch.long)))
            window = window[1:]
            token = tokenizer.id_to_token(new_token)
            print(f"Token '{token}' with id {new_token}")
            tokens.append(new_token)
        print("--- OUTPUT ---")
        print(gptok.metaspace_cleanup(tokenizer.decode(tokens))[1:])
