# before using this script, please run:
#       go install github.com/stretchr/testify@latest
#       go mod init testify
#       go mod tidy

import argparse
import logging
import random
import subprocess
import json
from typing import List

import numpy as np
from model import GopilotModel, GopilotConfig
import torch
from transformers.generation import StoppingCriteria, StoppingCriteriaList
from transformers import GenerationConfig
from tokenizer.tokenizer import GopilotTokenizer, HuggingFaceTokenizer, Tokenizer
import os
import os.path


class StopTokensStoppingCriteria(StoppingCriteria):
    def __init__(self, pad_token_id: int, stop_ids: List[int]):
        self.pad_token_id = pad_token_id
        self.stop_ids = stop_ids

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor):
        # Starting from the right, find the first index that isn't a pad_token_id token.
        # This is the index of the last token in the sequence.
        sequence = input_ids[0]
        for i in range(len(sequence) - 1, -1, -1):
            if sequence[i] != self.pad_token_id:
                break
        last_token_index = i
        # Check whether the last token is a stop token.
        if sequence[last_token_index] in self.stop_ids:
            return True
        return False


def post_processing(sequence: str):
    """
    Cleans up model generated sequences by trimming the string to the original function definition.

    Example:
        Input:
    ```
        return x + y
    }
    
    func Sub(x int, y int) int {
        return x - y
    }
    ```

        Output:
    ```
        return x + y
    }
    ```
    """
    index = sequence.find("\n}")
    if index != -1:
        return sequence[:index+2]
    else:
        return sequence


def evaluate_humanevalx_pass_at_k(tokenizer: Tokenizer, model: GopilotModel, k=100, max_new_tokens: int = 256, verbose: bool = False):
    evaluation_summary = {"tasks": [], f"pass@{k}": 0}

    with open("dataset/evaluation/humanevalx_go.jsonl", 'r') as f:
        # file is a list of dicts.
        # each dict has these keys: 'task_id', 'prompt', 'import', 'docstring', 'declaration', 'canonical_solution', 'test', 'test_setup', 'example_test'
        # import, docstring, declaration are contained in prompt. 
        humanevalx_samples = [json.loads(l) for l in f.readlines()]
    humanevalx_samples = [(jd['task_id'], jd['prompt'], jd['test_setup'], jd['test']) for jd in humanevalx_samples]

    completions = []

    pad_token_id = tokenizer.special_token_to_id("[PAD]")
    sc = StoppingCriteriaList([
        StopTokensStoppingCriteria(pad_token_id, [tokenizer.special_token_to_id("[EOS]")]),
    ])

    logging.info(f"Generating completions for {len(humanevalx_samples)} tasks")
    with torch.no_grad():
        for task_id, next_prompt, test_setup, test in humanevalx_samples:
            if verbose:
                logging.info(f"Generating completions for task {task_id}")
            prompt_tokens = tokenizer.encode(next_prompt)
            constrained_max_new_tokens = min(max_new_tokens, model.get_config().context_length - len(prompt_tokens))
            inputs = torch.tensor(prompt_tokens).long().unsqueeze(0).to(model.device)
            outputs: torch.Tensor = model.generate(
                inputs,
                pad_token_id=pad_token_id,
                max_new_tokens=constrained_max_new_tokens,
                temperature=0.5,
                do_sample=True,
                top_k=50,
                num_return_sequences=k,
                stopping_criteria=sc
            )
            assert outputs.shape[0] == k, f"outputs.shape: {outputs.shape}"
            candidate_sequences = [candidate_sequence.tolist()[len(prompt_tokens):] for candidate_sequence in outputs]
            assert len(candidate_sequences) == k, f"len(candidate_sequences): {len(candidate_sequences)}"
            assert len(candidate_sequences[0]) == max_new_tokens, f"len(candidate_sequences[0]): {len(candidate_sequences[0])}"
            candidate_sequences = [tokenizer.decode(candidate_sequence) for candidate_sequence in candidate_sequences]
            # For each candidate_sequence, we apply the post-processing heuristic,
            # to remove any extra code after the single function definition.
            candidate_sequences = [post_processing(candidate_sequence) for candidate_sequence in candidate_sequences]
            completions.append((task_id, next_prompt, test_setup, test, candidate_sequences))

    logging.info(f"Evaluating correctness of completions ({len(completions) * k} sequences)")
    for task_id, prompt, test_setup, test, candidate_sequences in completions:
        test_file_contents = test_setup + '\n' + test
        with open("evaluate_test.go", 'w') as f:
            f.write(test_file_contents)

        candidate_sequences_summary = []

        for candidate_id, candidate_sequence in enumerate(candidate_sequences):
            source_file_contents = "package main\n\n" + prompt + candidate_sequence + '\n\nfunc main() {}'

            with open("evaluate.go", "w") as f:
                f.write(source_file_contents)

            has_passed = False
            has_compiled = False
            completed_process = subprocess.run(["go", "build", "."], stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
            if completed_process.returncode == 0:
                has_compiled = True
                completed_process = subprocess.run(["go", "test", "."], stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
                if completed_process.returncode == 0:
                    has_passed = True

            if verbose:
                logging.info(f"Task {task_id} | PASS - {has_passed}, COMPILE - {has_compiled}")

            candidate_sequences_summary.append({
                "compiled": has_compiled,
                "passed": has_passed,
                "error_log": completed_process.stderr.decode('utf-8') if completed_process.returncode != 0 else None,
                "code": source_file_contents,
            })

        num_compiled = 0
        num_passed = 0
        for candidate_sequence_summary in candidate_sequences_summary:
            if candidate_sequence_summary["compiled"]:
                num_compiled += 1
            if candidate_sequence_summary["passed"]:
                num_passed += 1

        evaluation_summary["tasks"].append({
            "task_id": task_id,
            "candidate_sequences": candidate_sequences_summary,
            "num_compiled": num_compiled,
            "num_passed": num_passed,
        })

        logging.info(f"Task {task_id} | Pass Rate - {num_passed / k} | Compile Rate - {num_compiled / k}")

    for file in ["evaluate.go", "evaluate_test.go"]:
        if os.path.exists(file):
            os.remove(file)

    evaluation_summary[f"pass@{k}"] = sum(1 for task in evaluation_summary["tasks"] if task["num_passed"] > 0) / len(evaluation_summary["tasks"])
    evaluation_summary[f"compile@{k}"] = sum(1 for task in evaluation_summary["tasks"] if task["num_compiled"] > 0) / len(evaluation_summary["tasks"])

    # Write evaluation summary to a file.
    with open("evaluation_summary.json", 'w') as f:
        f.write(json.dumps(evaluation_summary, indent=4))

    return evaluation_summary

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
    # General arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-cf', type=str, required=True, help='Path to the model configuration file.')
    parser.add_argument('--tokenizer-cf', type=str, required=True, help='Path to the tokenizer configuration file.')
    parser.add_argument('--tokenizer', type=str, default="hugging-face", help='Name of the tokenizer to use.', choices=["gopilot", "hugging-face"])
    parser.add_argument('--model-weights', type=str, default=None, help='Path to the model weights.')
    parser.add_argument('--device', type=str, default="cuda", help='Device to use.', choices=["cuda", "cpu", "mps"])
    parser.add_argument('--k', type=int, default=10, help='Number of completions to generate for each task.')
    parser.add_argument('--verbose', action='store_true', help='Whether to print verbose logs.')
    parser.add_argument('--max-new-tokens', type=int, default=128, help='Maximum number of tokens to generate for each completion.')
    args = parser.parse_args()
    
    # Seed for reproducibility
    torch.manual_seed(999)
    np.random.seed(999)
    random.seed(999)

    # Model
    model = GopilotModel.from_config_file(args.model_cf, 0.0)
    
    # Load model from checkpoint
    checkpoint = torch.load(args.model_weights, map_location=args.device)
    for key in list(checkpoint['model'].keys()):
        if key.startswith("_orig_mod."):
            checkpoint['model'][key[len("_orig_mod."):]] = checkpoint['model'].pop(key)
    model.load_state_dict(checkpoint['model'])

    # Load the tokenizer
    if args.tokenizer == "gopilot":
        tokenizer = GopilotTokenizer.from_file(args.tokenizer_cf)
    else:
        tokenizer = HuggingFaceTokenizer.from_file(args.tokenizer_cf)

    logging.info(f"Evaluating...")    
    evaluate_humanevalx_pass_at_k(tokenizer, model, k=args.k, max_new_tokens=args.max_new_tokens, verbose=args.verbose)
