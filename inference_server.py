# Runs an inference server where HTTP requests can be made to a POST /complete
# endpoint to generate a completion for a given prompt.

import json
import argparse
from http.server import BaseHTTPRequestHandler, HTTPServer
import logging


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
        # generated_tokens = your_model.forward_pass(content, cursor_offset)

        self._send_response(200,json.dumps({"tokens": generated_tokens}))

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, default=3000)
    args = parser.parse_args()
    httpd = HTTPServer(('localhost', args.port), InferenceServer)
    logging.info(f"Starting inference server on port {args.port}")
    httpd.serve_forever()
