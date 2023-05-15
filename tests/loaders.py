import logging
import os
import sys
from typing import List
import unittest

# Add the parent directory to the path so we can import the module
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from dataset import DataPipeline, CachedS3DataSource, ParquetExtractorWithTokenization, StridedWindowBatcher
from tokenizer import GoScannerTokenizer


class TestLoaders(unittest.TestCase):
    def test_s3_parquet_with_tokenization_and_strided_window_batcher(self):
        tokenizer = GoScannerTokenizer.from_file(".cache/tokenizers/the-stack-dedup-v1.2/go-scanner-bpe-base/tokenizer.json")
        source = CachedS3DataSource(bucket="gopilot", cache_dir=".cache", file_lambda=lambda x: x.endswith(".parquet"), prefix="datasets/the-stack-dedup-v1.2/base")
        extractor = ParquetExtractorWithTokenization(transform=tokenizer.encode)
        batcher = StridedWindowBatcher(batch_size=16, window_size=128, stride=64)
        loader = DataPipeline(source, extractor, batcher)

        logging.info("Running over the 1000 first batches")
        for i, batch in enumerate(loader):
            self.assertEqual(batch.shape, (16, 128))
            if i == 1000:
                break


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    unittest.main()
