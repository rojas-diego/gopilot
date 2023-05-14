# Runs an inference server where HTTP requests can be made to a POST /complete
# endpoint to generate a completion for a given prompt.

import json
import argparse
from http.server import BaseHTTPRequestHandler, HTTPServer
import logging

import torch

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
        generated_tokens = forward_pass(content, cursor_offset)
        self._send_response(200,json.dumps({"tokens": generated_tokens}))

model = None

def forward_pass(content, cursor_offset):
    if model is None:
        return ""
    context_length = model.get_config().context_length
    before = content[:cursor_offset]
    tokenized_before = tokenizer.encode(before)
    tokenized_before = tokenized_before[-context_length:]
    left_padded_tokenized_before = ([tokenizer.special_token_to_id("[PAD]")] * (context_length - len(tokenized_before))) + tokenized_before
    outputs = model.forward(torch.tensor([left_padded_tokenized_before]))
    predicted_token_id = torch.argmax(outputs.logits[0, -1, :]) # type: ignore
    predicted_token = tokenizer.id_to_token(int(predicted_token_id.item()))
    return predicted_token


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, default=3000)
    parser.add_argument('--model', type=str, required=True, help="Path to model configuration")
    parser.add_argument('--tokenizer', type=str, required=True, help="Path to tokenizer configuration")
    parser.add_argument('--checkpoint', type=str, required=True, help="Path to model checkpoint")
    args = parser.parse_args()

    model = GopilotModel.from_config_file(args.model, dropout=0.0)
    tokenizer = GoScannerTokenizer.from_file(args.tokenizer)
    checkpoint = torch.load(args.checkpoint)
    model.load_state_dict(checkpoint['model'])

    httpd = HTTPServer(('localhost', args.port), InferenceServer)
    logging.info(f"Starting inference server on port {args.port}")
    httpd.serve_forever()
