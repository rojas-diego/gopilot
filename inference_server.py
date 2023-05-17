# Runs an inference server where HTTP requests can be made to a POST /complete
# endpoint to generate a completion for a given prompt.

import json
import argparse
from http.server import BaseHTTPRequestHandler, HTTPServer
import logging

import torch
import flame

from model.model import GopilotModel
from tokenizer.tokenizer import GoScannerTokenizer


class InferenceServer(BaseHTTPRequestHandler):
    def _send_response(self, status_code, body):
        self.send_response(status_code)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'POST')
        self.end_headers()
        self.wfile.write(body.encode())

    def do_OPTIONS(self):
        self.send_response(200, "ok")
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'POST')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.end_headers()

    def do_POST(self):
        content_length = int(self.headers['Content-Length'])
        post_data = self.rfile.read(content_length)
        data = json.loads(post_data)
        content = data.get("content")
        cursor_offset = data.get("cursorOffset")
        if content is None or cursor_offset is None:
            self._send_response(400, json.dumps({"error": "Invalid request body"}))
            return
        generated_tokens = "Example generated tokens"
        logging.info(f"Received request with content length {len(content)} and cursor offset {cursor_offset}")
        # Your model's forward pass implementation here
        generated_tokens = generate(content, cursor_offset, 3)
        self._send_response(200,json.dumps({"tokens": generated_tokens}))

model = None

def generate(content, cursor_offset, max_tokens):
    if model is None:
        return ""
    context_length = model.get_config().context_length
    before = content[:cursor_offset]
    tokenized_before = tokenizer.encode(before)
    if len(tokenized_before) > context_length:
        tokenized_before = tokenized_before[-context_length:]
    right_padded_tokenized_before = tokenized_before + ([tokenizer.special_token_to_id("[PAD]")] * (context_length - len(tokenized_before)))
    print("Tokenized Before:", [tokenizer.id_to_token_name(id) for id in right_padded_tokenized_before])

    generated_token_ids = []
    while len(generated_token_ids) < max_tokens:
        inputs = torch.tensor([right_padded_tokenized_before])
        attention_mask = (inputs != tokenizer.special_token_to_id("[PAD]")).long()
        outputs = model(inputs, attention_mask=attention_mask)
        predicted_token_id = torch.argmax(outputs.logits[0, -1, :]) # type: ignore
        predicted_token = tokenizer.id_to_token(int(predicted_token_id.item()))
        generated_token_ids.append(predicted_token_id.item())
        right_padded_tokenized_before = right_padded_tokenized_before[1:] + [predicted_token_id.item()]
        print("Generating...")

    print("Predicted Tokens:", [tokenizer.id_to_token_name(token_id) for token_id in generated_token_ids])
    decoded_prediction = tokenizer.decode(generated_token_ids)
    print("Predicted String:", decoded_prediction)
    return decoded_prediction


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, default=3000)
    parser.add_argument('--model', type=str, required=True, help="Path to model configuration")
    parser.add_argument('--tokenizer', type=str, required=True, help="Path to tokenizer configuration")
    parser.add_argument('--checkpoint', type=str, required=True, help="Path to model checkpoint")
    parser.add_argument('--device', type=str, help="Device to run inference on")
    args = parser.parse_args()

    args.device = torch.device(args.device) if args.device else flame.best_device()

    model = GopilotModel.from_config_file(args.model, dropout=0.0)
    tokenizer: GoScannerTokenizer = GoScannerTokenizer.from_file(args.tokenizer) # type: ignore
    checkpoint = torch.load(args.checkpoint, map_location=args.device)
    # For every key in checkpoint that begins with `_orig_mod.`, remove that prefix
    # and load the state dict into the model.
    for key in list(checkpoint['model'].keys()):
        if key.startswith("_orig_mod."):
            checkpoint['model'][key[len("_orig_mod."):]] = checkpoint['model'].pop(key)
    model.load_state_dict(checkpoint['model'])

    httpd = HTTPServer(('localhost', args.port), InferenceServer)
    logging.info(f"Starting inference server on port {args.port}")
    httpd.serve_forever()
