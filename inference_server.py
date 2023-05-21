"""
This file contains an example inference server that can be used to serve code
completions. It is used by the VSCode extension to serve code completions.
"""

import argparse
import json
import logging
from concurrent.futures import ThreadPoolExecutor, Future
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Callable, List
import uuid
from transformers.generation import StoppingCriteria, StoppingCriteriaList

import torch

import flame
from model.model import GopilotModel
from tokenizer import GopilotTokenizer, HuggingFaceTokenizer, Tokenizer


@dataclass
class CompletionTask:
    id: str
    file_contents: str
    cursor_offset: int
    max_new_tokens: int
    stopping_tokens: List[int] # TODO: Implement

    def __str__(self):
        return f"CompletionTask(id={self.id})"


def tokenize_file_context(tokenizer: Tokenizer, file_contents: str, cursor_offset: int, context_length: int, max_new_tokens: int):
    """
    Tokenizes a file and truncates it to the appropriate context length.

    Args:
        `tokenizer`: The tokenizer to use to tokenize the file.
        `file_contents`: The contents of the file to be tokenized.
        `cursor_offset`: The offset of the cursor in the file.
        `context_length`: The context length to pad the file to.
        `pad_token_id`: The ID of the pad token.
        `max_new_tokens`: Max tokens that will be generated. Used to truncate.

    Returns:
        A tensor containing the tokenized file, padded to the context length.
    """
    assert context_length >= max_new_tokens
    file_context = tokenizer.encode(file_contents[:cursor_offset])
    if len(file_context) > context_length - max_new_tokens:
        file_context = file_context[-(context_length - max_new_tokens):]
    print([tokenizer.id_to_token(token_id) for token_id in file_context])
    return torch.tensor(file_context)


class ModelService:
    """Allows completion tasks to be queued and processed by a model."""
    def __init__(self, tokenizer: Tokenizer, model: GopilotModel, temperature: float = 1.0, repetition_penalty: float = 1.2, max_length: int = 3):
        self._model = model
        self._executor = ThreadPoolExecutor(max_workers=1)
        self._tokenizer = tokenizer
        self._temperature = temperature
        self._repetition_penalty = repetition_penalty
        self._max_length = max_length
        self._model_context_length = model.get_config().context_length
        self._pad_token_id = tokenizer.special_token_to_id("[PAD]")

    def queue_completion_task(self, task: CompletionTask) -> Future:
        """
        Queues a completion task to be executed by the model.

        Args:
            `task`: The completion task to be executed.

        Returns:
            A future that can be used to retrieve the result of the completion
            task.
        """
        logging.info(f"Queuing completion task: {task.id}")
        return self._executor.submit(self._new_completion_task_handler(task))

    def _new_completion_task_handler(self, task: CompletionTask) -> Callable[[], str]:
        def completion_task_handler():
            logging.info(f"Executing completion task: {task.id}")
            input_ids = tokenize_file_context(
                self._tokenizer,
                task.file_contents,
                task.cursor_offset,
                self._model_context_length,
                task.max_new_tokens)
            assert input_ids.numel() <= self._model_context_length
            logging.info(f"Feeding model {input_ids.numel()} tokens")
            input_ids = input_ids.unsqueeze(0)
            output = self._model.generate(
                input_ids,
                do_sample=True,
                temperature=self._temperature,
                max_new_tokens=min(task.max_new_tokens, self._model_context_length - input_ids.numel()),
                repetition_penalty=self._repetition_penalty,
                pad_token_id=self._pad_token_id)
            assert isinstance(output, torch.Tensor)
            decoded_output = self._tokenizer.decode(output.squeeze(0).tolist()[input_ids.numel():])
            logging.info(f"Completion task {task.id} completed.")
            return decoded_output

        return completion_task_handler


def new_inference_endpoint_handler(model_service: ModelService, tokenizer: Tokenizer):
    class InferenceEndpointHandler(BaseHTTPRequestHandler):
        def _send_response(self, status_code, body):
            self.send_response(status_code)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.send_header('Access-Control-Allow-Methods', 'POST')
            self.end_headers()
            self.wfile.write(body.encode())

        def _read_json_body(self):
            content_length = int(self.headers['Content-Length'])
            post_data = self.rfile.read(content_length)
            return json.loads(post_data)
        
        def _new_id(self):
            return str(uuid.uuid4())

        def do_OPTIONS(self):
            self.send_response(200, "ok")
            self.send_header('Access-Control-Allow-Origin', '*')
            self.send_header('Access-Control-Allow-Methods', 'POST')
            self.send_header('Access-Control-Allow-Headers', 'Content-Type')
            self.end_headers()

        def do_POST(self):
            body = self._read_json_body()
            completion_task = CompletionTask(
                id=self._new_id(),
                file_contents=body.get("fileContents"),
                cursor_offset=body.get("cursorOffset"),
                max_new_tokens=12,
                stopping_tokens=[]
            )
            result = model_service.queue_completion_task(completion_task).result()
            self._send_response(200, json.dumps({"tokens": result}))

    return InferenceEndpointHandler


def load_model_from_checkpoint(model_cf: str, checkpoint_path: str, device: torch.device):
    logging.info(f"Loading model from checkpoint '{checkpoint_path}'")
    model = GopilotModel.from_config_file(model_cf)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    # For every key in checkpoint that begins with `_orig_mod.`, remove that prefix
    # and load the state dict into the model.
    for key in list(checkpoint['model'].keys()):
        if key.startswith("_orig_mod."):
            checkpoint['model'][key[len("_orig_mod."):]] = checkpoint['model'].pop(key)
    model.load_state_dict(checkpoint['model'])
    return model


def load_tokenizer_from_file(tokenizer: str, tokenizer_cf: str):
    logging.info(f"Loading tokenizer from '{tokenizer_cf}'")
    if tokenizer == "gopilot":
        return GopilotTokenizer.from_file(tokenizer_cf)
    else:
        return HuggingFaceTokenizer.from_file(tokenizer_cf)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, default=3000)
    parser.add_argument('--tokenizer', type=str, required=True, choices=["gopilot", "hugging-face"], help="Name of the tokenizer.")
    parser.add_argument('--model-cf', type=str, required=True, help="Path to model configuration")
    parser.add_argument('--tokenizer-cf', type=str, required=True, help="Path to tokenizer configuration")
    parser.add_argument('--checkpoint-path', type=str, required=True, help="Path to model checkpoint")
    parser.add_argument('--device', type=str, help="Device to run inference on")
    args = parser.parse_args()

    args.device = torch.device(args.device) if args.device else flame.best_device()

    tokenizer = load_tokenizer_from_file(args.tokenizer, args.tokenizer_cf)
    model = load_model_from_checkpoint(args.model_cf, args.checkpoint_path, args.device)

    flame.log_model_summary(model)
    logging.info(f"Tokenizer vocabulary size: {tokenizer.get_vocab_size()}")

    httpd = HTTPServer(('localhost', args.port), new_inference_endpoint_handler(ModelService(tokenizer, model), tokenizer))
    logging.info(f"Starting inference server on port {args.port}")
    httpd.serve_forever()
