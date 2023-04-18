import logging

import tiktoken
import torch
import torch.nn as nn
from torchtext.datasets import WikiText2

import dlutils

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")


TOKENIZER = tiktoken.encoding_for_model("gpt-4")

logging.info(f"Using vocabulary of size {TOKENIZER.max_token_value + 1}")

PADDING_TOKEN = '<pad>'
SEP_TOKEN = '<sep>'
EMBEDDING_DIMENSIONS = 100
BATCH_SIZE = 128


class LSTMLanguageModel(nn.Module):
    def __init__(self, vocab_size: int, embedding_dim: int, hidden_size: int, num_layers: int, dropout=0.5):
        super(LSTMLanguageModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_size, num_layers, dropout=dropout, batch_first=True)
        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, x, hidden):
        embedded = self.embedding(x)
        lstm_output, hidden = self.lstm(embedded, hidden)
        output = self.fc(lstm_output)
        return output, hidden

    def init_hidden(self, batch_size, device):
        return (torch.zeros(self.lstm.num_layers, batch_size, self.lstm.hidden_size).to(device),
                torch.zeros(self.lstm.num_layers, batch_size, self.lstm.hidden_size).to(device))


train_datapipe, validation_datapipe, test_datapipe = WikiText2(split=('train', 'valid', 'test'), root='data')  # type: ignore


train_stream = dlutils.StreamDataLoader(
    train_datapipe,
    batch_size=BATCH_SIZE,
    tokenizer_fn=TOKENIZER.encode,
    pad_token_id=TOKENIZER.encode(PADDING_TOKEN)[0],
    sep_token_id=TOKENIZER.encode(SEP_TOKEN)[0],
    max_sequence_len=128,
    buffer_size=10
)

model = LSTMLanguageModel(TOKENIZER.max_token_value + 1, EMBEDDING_DIMENSIONS, 128, 2)

dlutils.xavier_initialization(model)

task = dlutils.SupervisedLanguageModelingTask(
    model,
    torch.nn.BCEWithLogitsLoss(),
    torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=0.0001),
)

trainer = dlutils.Trainer(
    task,
    dlutils.best_device(),
    train_datapipe,
    test_datapipe,
    test_datapipe,
)
trainer.register_handlers(
    dlutils.LoggingHandler(),
)
trainer.train(10)
trainer.test()
