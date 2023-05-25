import logging
import os
from typing import List
import numpy

import pandas
import tqdm

from tokenizer import GopilotTokenizer
from tokenizers import Tokenizer

from .preprocessing_job import PreprocessingJob
import multiprocessing
import functools


class TokenizeWithHuggingFaceJob(PreprocessingJob):
    def run(self):
        """Tokenizes the dataset using the given tokenizer. For each chunk of
        the dataset, produce a .npy file containing a single continguous token
        array."""
        super().run()
        total_num_tokens = 0
        self.tokenizer: Tokenizer = Tokenizer.from_file("tokenizer/config/hugging-face.json")
        for file in self.files():
            df = pandas.read_parquet(file)
            batch_tokens = self.tokenizer.encode_batch(df["content"])
            batch_ids = [token.ids for token in batch_tokens]
            num_tokens = sum([len(ids) for ids in batch_ids])
            total_num_tokens += num_tokens
            ids = numpy.empty(num_tokens, dtype=numpy.uint16)
            offset = 0
            for token_ids in batch_ids:
                ids[offset:offset + len(token_ids)] = token_ids
                offset += len(token_ids)
            self.save_npy(ids, os.path.basename(file).replace('.parquet', '.npy'))
        self.save_json({"num_tokens": total_num_tokens}, "summary.json")


class TokenizeWithGopilotJob(PreprocessingJob):
    def __init__(self, **kwargs):
        super(TokenizeWithGopilotJob, self).__init__(**kwargs)
        logging.info('Starting TokenizeWithGopilotJob')

    def run(self):
        """For each chunk of the dataset, produce a .npy file containing a
        single continguous token array."""
        super().run()
        logging.info('run called')
        total_num_tokens = 0
        self.tokenizer = GopilotTokenizer.from_file("tokenizer/config/gopilot.json")
        for file in self.files():
            logging.info(f'TOKENIZING this file: {file}')
            df = pandas.read_parquet(file, columns=['content'])

            with multiprocessing.Pool() as p:
                batch_ids_uncollapsed = p.map(self.tokenizer.encode_batch, map(
                    lambda i: df["content"][int(i*len(df)/100):int((i+1)*len(df)/100)],
                    range(100)
                ))
            batch_ids = functools.reduce(lambda a,b: a+b, batch_ids_uncollapsed)

            num_tokens = sum([len(ids) for ids in batch_ids])
            total_num_tokens += num_tokens
            ids = numpy.empty(num_tokens, dtype=numpy.uint16)
            offset = 0
            for token_ids in batch_ids:
                ids[offset:offset + len(token_ids)] = token_ids
                offset += len(token_ids)
            self.save_npy(ids, os.path.basename(file).replace('.parquet', '.npy'))
        self.save_json({"num_tokens": total_num_tokens}, "summary.json")
