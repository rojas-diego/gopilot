from dataclasses import dataclass


@dataclass
class TokenizeSummary:
    num_samples: int = 0
    num_tokens: int = 0


class Job:
    def __init__(self):
        self.tokenize_summary = TokenizeSummary(
            num_samples=0,
            num_tokens=0,
        )

    def run(self, infile: str, outfile: str):
        raise NotImplementedError

    def summary(self) -> dict:
        return self.tokenize_summary.__dict__
