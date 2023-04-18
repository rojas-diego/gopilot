# Sample script to examine how tiktoken works
import tiktoken

# Load the tokenizer from user input
tokenizer = tiktoken.encoding_for_model(input("Enter model name: "))

while True:
    # Tokenize the input
    tokens = tokenizer.encode(input("Enter text: "))
    print(tokens)
