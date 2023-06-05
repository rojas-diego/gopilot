# before using this script, please run:
#       go install github.com/stretchr/testify@latest
#       go mod init testify
#       go mod tidy

import subprocess
import json
from pdb import set_trace as st
from model import GopilotModel, GopilotConfig
import torch
from transformers.generation import StoppingCriteria, StoppingCriteriaList
from transformers import GenerationConfig
from tokenizer.tokenizer import HuggingFaceTokenizer
import os
import os.path
import re

class SC(StoppingCriteria):
    FUNC_ID = 32747
    def __init__(self):
        super().__init__()

    def __call__(self, input_ids, scores):
        nf = 0
        for id in input_ids[0]:
            if id == self.FUNC_ID:
                nf+=1
        return nf >= 2

# tok: the tokenizer itself. not the config file
# model: the model itself. not the config file
# prompt_jsonl_path: path to humanevalx_go.jsonl
def autocomplete_with_model(tok, model, prompt_jsonl_path='dataset/evaluation/humanevalx_go.jsonl', num_beams=200, num_return_sequences=200, max_new_tokens=100, samples_out_file='samples.jsonl'):
    with open(prompt_jsonl_path, 'r') as f:
        # file is a list of dicts.
        # each dict has these keys: 'task_id', 'prompt', 'import', 'docstring', 'declaration', 'canonical_solution', 'test', 'test_setup', 'example_test'
        # import, docstring, declaration are contained in prompt. 
        json_data = [json.loads(l) for l in f.readlines()]
    
    inputs = [(jd['task_id'], jd['prompt'], jd['test_setup'], jd['test']) for jd in json_data]

    sc = StoppingCriteriaList([SC()])
    
    for task_id, next_prompt, test_setup, test in inputs:
        x = torch.Tensor(tok.encode(next_prompt)).int().unsqueeze(0)
        y = model.generate(x, attention_mask=torch.ones(x.shape), max_new_tokens=max_new_tokens, num_beams=num_beams, num_return_sequences=num_return_sequences, stopping_criteria=sc)
        completions = [tok.decode(list(y_))[len(next_prompt):] for y_ in y]

        with open(samples_out_file, 'a') as f:
            for c in completions:
                f.write(json.dumps(
                    {
                        'task_id': task_id,
                        'prompt': next_prompt,
                        'completion': c.strip(),
                        'test_setup': test_setup,
                        'test': test
                    }
                ) + '\n')

def _get_imports(string):
    string = string[string.find('import')+6:].strip()
    if string[0] == '"':
        string = string[1:]
        string = string[:string.find('"')]
        return [string.strip()]
    else:
        string = string[1:]
        string = string[:string.find(')')]
        ims = string.split('\n')
        rv = []
        for im in ims:
            if im.strip() == '':
                continue
            else:
                rv.append(im.strip()[1:-1])
        return rv

def evaluate_model_samples(samples_file='samples.jsonl', test_go_file='testing_test.go'):
    with open(samples_file, 'r') as f:
        json_data = [json.loads(l) for l in f.readlines()]

    task_id_pass_counts = dict()
    
    for jd in json_data:
        task_id = jd['task_id']
        prompt = jd['prompt']
        completion = jd['completion']
        test_setup = jd['test_setup']
        test = jd['test']

        if test_setup.find('import') != -1 and prompt.find('import') != -1:
            test_setup_imports = _get_imports(test_setup)
            prompt_imports = _get_imports(prompt)
            if not set(test_setup_imports).isdisjoint(set(prompt_imports)):
                for pi in prompt_imports:
                    test_setup = test_setup.replace(f'"{pi}"', '')

        test_code = test_setup + '\n' + prompt + '\n' + completion + '\n' + test

        with open(test_go_file, 'w') as f:
            f.write(test_code)

        task_id_pass_counts[task_id] = 0
        if 'PASS' in subprocess.run(['go', 'test'], stdout=subprocess.PIPE).stdout.decode('utf-8'):
            task_id_pass_counts[task_id] += 1
            
    if os.path.exists(test_go_file):
        os.remove(test_go_file)

    return task_id_pass_counts

if __name__ == '__main__':

    gpc = GopilotConfig(context_length=512,
        embedding_dimensions=4,
        num_layers=2,
        num_heads=4,
        feedforward_dimensions=512,
        vocab_size=32768
    )
    
    model = GopilotModel(gpc)
    tok = HuggingFaceTokenizer.from_file('tokenizer/config/hugging-face.json')
    autocomplete_with_model(tok, model, num_beams=2, num_return_sequences=2,max_new_tokens=10)

    # x = evaluate_model_samples('samples.jsonl')
    # print(x)