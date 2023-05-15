from .batchers import Batcher
from .extractors import Extractor
from .sources import DataSource


class DataPipeline:
    def __init__(self, data_source: DataSource, extractor: Extractor, batcher: Batcher):
        self.data_source = data_source
        self.extractor = extractor
        self.batcher = batcher

    def __iter__(self):
        for batch in self.batcher.batches(self.extractor.samples(self.data_source.files())):
            yield batch
